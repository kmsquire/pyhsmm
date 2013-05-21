from __future__ import division
from pybasicbayes.models import *

from ..util.stats import sample_discrete_from_log

# this FrozenMixtureDistribution leaves its components' parameters fixed and
# only adjusts its weights

import os
eigen_path = os.path.join(os.path.dirname(__file__),'../deps/Eigen3')

class FrozenLabels(Labels):
    def __init__(self,likelihoods,*args,**kwargs):
        super(FrozenLabels,self).__init__(*args,**kwargs)
        self._likelihoods = likelihoods

    def meanfieldupdate(self):
        raise NotImplementedError

    def resample(self,temp=None):
        scores = self._likelihoods[self.data] + \
                self.weights.log_likelihood(np.arange(len(self.components)))

        if temp is not None:
            scores /= temp

        self.z = sample_discrete_from_log(scores,axis=1)

    def E_step(self):
        data, N, K = self.data, self.data.shape[0], len(self.components)
        self.expectations = np.empty((N,K))

        self.expectations = self._likelihoods[data]

        self.expectations += self.weights.log_likelihood(np.arange(K))

        self.expectations -= self.expectations.max(1)[:,na]
        np.exp(self.expectations,out=self.expectations)
        self.expectations /= self.expectations.sum(1)[:,na]

        self.z = self.expectations.argmax(1)

class FrozenMixtureDistribution(MixtureDistribution):
    @staticmethod
    def get_all_likelihoods(components,data):
        if not isinstance(data,np.ndarray):
            raise NotImplementedError

        likelihoods = np.empty((data.shape[0],len(components)))
        for idx, c in enumerate(components):
            likelihoods[:,idx] = c.log_likelihood(data)
        maxes = likelihoods.max(axis=1)
        shifted_likelihoods = np.exp(likelihoods - maxes[:,na])
        return likelihoods, maxes, shifted_likelihoods

    def __init__(self,all_data,all_likelihoods,maxes,shifted_likelihoods,*args,**kwargs):
        super(FrozenMixtureDistribution,self).__init__(*args,**kwargs)
        self._likelihoods = all_likelihoods
        self._maxes = maxes
        self._shifted_likelihoods = shifted_likelihoods
        self._data = all_data

    def add_data(self,data):
        # NOTE: data is indices
        self.labels_list.append(FrozenLabels(
            data=data.astype(np.int64),
            components=self.components,
            weights=self.weights,
            likelihoods=self._likelihoods))

    def resample(self,data,niter=5,temp=None):
        if getdatasize(data) > 0:
            if not isinstance(data,np.ndarray):
                data = np.concatenate(data)

            self.add_data(data)

            for itr in range(niter):
                self.resample_model(temp=temp)
            self.weights.resample([l.z for l in self.labels_list])

            self.labels_list.pop()
        else:
            self.resample_model(temp=temp)

    def resample_model(self, temp=None):
        for l in self.labels_list:
            l.resample(temp=temp)
        if hasattr(self.weights,'resample_just_weights'):
            self.weights.resample_just_weights([l.z for l in self.labels_list]) # don't do concentration
        else:
            self.weights.resample([l.z for l in self.labels_list])

    def log_likelihood(self,x):
        return self.log_likelihood_faster_2(x)

    def log_likelihood_slower(self,x):
        # NOTE: x is indices
        K = len(self.components)
        vals = self._likelihoods[x.astype(np.int64)]
        vals += self.weights.log_likelihood(np.arange(K))
        return np.logaddexp.reduce(vals,axis=1)

    def log_likelihood_faster(self,sub_indices):
        # NOTE: this method takes INDICES into the data
        log_likelihoods = self._likelihoods
        log_weights = np.log(self.weights.weights)

        K = self.weights.weights.shape[0]
        num_sub_indices = sub_indices.shape[0]
        num_indices = log_likelihoods.shape[0]

        out = np.empty(num_sub_indices)

        scipy.weave.inline(
                '''
                using namespace Eigen;

                Map<ArrayXd> elog_weights(log_weights,K);
                Map<ArrayXXd> eall_log_likelihoods(log_likelihoods,K,num_indices);

                ArrayXd row(K);

                for (int i=0; i < num_sub_indices; i++) {
                    int idx = sub_indices[i];
                    row = elog_weights + eall_log_likelihoods.col(idx);
                    double themax = row.maxCoeff();
                    out[i] = log((row - themax).exp().sum()) + themax;
                }
                ''',['sub_indices','log_likelihoods','K','num_indices',
                    'num_sub_indices','out','log_weights'],
                headers=['<Eigen/Core>','<math.h>'],include_dirs=[eigen_path],
                extra_compile_args=['-O3','-DNDEBUG'])
        return out

    def log_likelihood_faster_2(self,sub_indices):
        # NOTE: this method takes INDICES into the data
        shifted_likelihoods = self._shifted_likelihoods
        maxes = self._maxes
        weights = self.weights.weights

        K = weights.shape[0]
        num_sub_indices = sub_indices.shape[0]
        num_indices = shifted_likelihoods.shape[0]

        out = np.empty(num_sub_indices)

        scipy.weave.inline(
                '''
                using namespace Eigen;

                Map<MatrixXd> eweights(weights,1,K);
                Map<MatrixXd> eall_likelihoods(shifted_likelihoods,K,num_indices);

                for (int i=0; i < num_sub_indices; i++) {
                    int idx = sub_indices[i];
                    out[i] = log((eweights * eall_likelihoods.col(idx)).array().value()) + maxes[idx];
                }
                ''',['sub_indices','shifted_likelihoods','K','num_indices',
                    'num_sub_indices','out','weights','maxes'],
                headers=['<Eigen/Core>','<math.h>'],include_dirs=[eigen_path],
                extra_compile_args=['-O3','-DNDEBUG'])
        return out

    def max_likelihood(self,data,weights=None):
        # NOTE: data is an array or list of arrays of indices
        if weights is not None:
            raise NotImplementedError
        assert isinstance(data,list) or isinstance(data,np.ndarray)
        if isinstance(data,np.ndarray):
            data = [data]

        if getdatasize(data) > 0:
            for d in data:
                self.add_data(d)

            for itr in range(10):
                self.EM_step()

            for d in data:
                self.labels_list.pop()

    def EM_step(self):
        assert all(isinstance(c,MaxLikelihood) for c in self.components), \
                'Components must implement MaxLikelihood'
        assert len(self.labels_list) > 0, 'Must have data to run EM'

        ## E step
        for l in self.labels_list:
            l.E_step()

        ## M step
        # mixture weights
        self.weights.max_likelihood(np.arange(len(self.components)),
                [l.expectations for l in self.labels_list])

    def plot(self,data=[],color='b',label='',plot_params=True):
        if not isinstance(data,list):
            data = [data]
        for d in data:
            self.add_data(d)

        for l in self.labels_list:
            l.E_step() # sets l.z to MAP estimates
            for label, o in enumerate(self.components):
                if label in l.z:
                    o.plot(color=color,label=label,
                            data=self._data[l.data[l.z == label]] if l.data is not None else None)

        for d in data:
            self.labels_list.pop()

    def __getstate__(self):
        return dict(weights=self.weights)

    def __setstate__(self,d):
        self.weights = d['weights'].astype(np.float64)
        # TODO need to set components library!

