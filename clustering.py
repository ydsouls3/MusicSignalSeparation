import sys
import soundfile as sf
import numpy as np
from scipy import signal as sg
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

class Clustering:

    def __init__(self, k_init, n_init, verbose, random_state, ic, alg):

        self.k_init = k_init
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.ic = ic
        self.alg = alg

    def fit(self, X):

        self.__clusters = [] 

        if self.alg == "kmeans":
            clusters = self.Cluster.build(X, self.alg, KMeans(n_clusters=self.k_init, n_init=self.n_init, verbose=self.verbose, random_state=self.random_state).fit(X))
        elif self.alg == "kshape":    
            clusters = self.Cluster.build(X, self.alg, KShape(n_clusters=self.k_init, n_init=self.n_init, verbose=self.verbose, random_state=self.random_state).fit(X))
        else:
            print("alg should be either 'kmeans' or 'kshape'")
            sys.exit()

        self.__recursively_split(clusters)

        self.labels_ = np.empty(X.shape[0], dtype = np.intp)
        for i, c in enumerate(self.__clusters):
            self.labels_[c.index] = i

        self.cluster_centers_ = np.array([c.center for c in self.__clusters])
        self.cluster_log_likelihoods_ = np.array([c.log_likelihood() for c in self.__clusters])
        self.cluster_sizes_ = np.array([c.size for c in self.__clusters])

        return self

    def __recursively_split(self, clusters):

        for cluster in clusters:
            if cluster.size <= 3:
                self.__clusters.append(cluster)
                continue
            if self.alg == "kmeans":
              k_cl = KMeans(n_clusters=self.k_init, n_init=self.n_init, verbose=self.verbose, random_state=self.random_state).fit(cluster.data)
            elif self.alg == "kshape":    
              k_cl = KShape(n_clusters=self.k_init, n_init=self.n_init, verbose=self.verbose, random_state=self.random_state).fit(cluster.data)


            c1, c2 = self.Cluster.build(cluster.data, self.alg, k_cl, cluster.index)

            beta = np.linalg.norm(c1.center - c2.center) / np.sqrt(np.linalg.det(c1.cov) + np.linalg.det(c2.cov))
            alpha = 0.5 / stats.norm.cdf(beta)
            
            if self.ic == "bic":
             bic = -2 * (cluster.size * np.log(alpha) + c1.log_likelihood() + c2.log_likelihood()) + 2 * cluster.df * np.log(cluster.size)
             if bic < cluster.bic():
                self.__recursively_split([c1, c2])
             else:
                self.__clusters.append(cluster)
            elif self.ic == "aic":
             aic = -2 * (cluster.size * np.log(alpha) + c1.log_likelihood() + c2.log_likelihood()) + 2 * cluster.df   
             if aic < cluster.aic():
                self.__recursively_split([c1, c2])
             else:
                self.__clusters.append(cluster)
            else:
             print("ic should be either 'bic' or 'aic'")
             sys.exit() 

    class Cluster:

        @classmethod
        def build(cls, X, alg, k_cl, index = None): 
            if index is None:
             index = np.array(range(0, X.shape[0]))
            labels = range(0, k_cl.get_params()["n_clusters"])  

            return tuple(cls(X, alg, index, k_cl, label) for label in labels) 

        # index: Xの各行におけるサンプルが元データの何行目のものかを示すベクトル
        def __init__(self, X, alg, index, k_cl, label):
            self.data = X[k_cl.labels_ == label]
            self.alg = alg
            self.index = index[k_cl.labels_ == label]
            self.size = self.data.shape[0]
            self.df = self.data.shape[1] * (self.data.shape[1] + 3) / 2
            self.center = k_cl.cluster_centers_[label]
            self.cov = np.cov(self.data.T)

        def log_likelihood(self):
            self.cov = np.where(np.isnan(self.cov),0,self.cov)
            dim = self.center.shape[0]
            self.center2 = np.empty(dim)
            if self.alg == "kmeans":
                log_likelihood = sum(stats.multivariate_normal.logpdf(x, self.center, self.cov, allow_singular=True) for x in self.data)
            elif self.alg == "kshape":
                for n in range(dim):
                 self.center2[n] = self.center[n,0]
                log_likelihood = sum(stats.multivariate_normal.logpdf(x, self.center2, self.cov, allow_singular=True) for x in self.data)
            return log_likelihood

        def bic(self):
            return -2 * self.log_likelihood() + 2 * self.df * np.log(self.size)
        
        def aic(self):
            return -2 * self.log_likelihood() + 2 * 2 * self.df   

def transform_vector(time_series_array):
    #ベクトルに変換
    stack_list = []
    for j in range(len(time_series_array)):
        data = np.array(time_series_array[j])
        data = data.reshape((1, len(data))).T
        stack_list.append(data)
    #一次元配列にする
    stack_data = np.stack(stack_list, axis=0)
    return stack_data

def zscore(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore

def get_nonzero(mat1, mat2):
    new_mat1 = np.where(mat1 < 1e-10, 1e-10, mat1)
    new_mat2 = np.where(mat2 < 1e-10, 1e-10, mat2)
    product = new_mat1 @ new_mat2
    return new_mat1, new_mat2, product

def get_NMF(Y, num_iter, num_base, loss_func):
    
    Y = np.where(Y < 1e-10, 1e-10, Y)
    
    #Initialize basements and activation based on the Y size(k, n)
    K, N = Y.shape[0], Y.shape[1]
    if num_base >= K or num_base >= N:
        print("The number of basements should be lower than input size.")
        sys.exit()
    H = np.random.rand(K, num_base) #basements
    H = H / H.sum(axis=0, keepdims=True) #Normalization
    U = np.random.rand(num_base, N) #activation
    
    #Initialize valuables
    loss = np.zeros(num_iter)
    HU = []
    
    #Repeat num_iter times
    for i in range(num_iter):
        
        #In the case of squared Euclidean distance
        if loss_func == "EU":
            #Update the basements
            H, U, X = get_nonzero(H, U)
            H = H * (Y @ U.T) / (X @ U.T)
            H = H / H.sum(axis=0, keepdims=True) #Normalization
            
            #Update the activation
            H, U, X = get_nonzero(H, U)
            U = U * (H.T @ Y) / (H.T @ X)
            
            #Compute the loss function
            H, U, X = get_nonzero(H, U)
            loss[i] = np.sum((Y - X)**2)
        
        #In the case of Kullback–Leibler divergence
        elif loss_func == "KL":
            #Update the basements
            H, U, X = get_nonzero(H, U)
            denom_H = U.T.sum(axis=0, keepdims=True) #(1xM) matrix
            H = H * ((Y / X) @ U.T) / denom_H
            H = H / H.sum(axis=0, keepdims=True) #Normalization
            
            #Update the activation
            H, U, X = get_nonzero(H, U)
            denom_U = H.T.sum(axis=1, keepdims=True) #(Mx1) matrix
            U = U * (H.T @ (Y / X)) / denom_U
            
            #Compute the loss function
            H, U, X = get_nonzero(H, U)
            loss[i] = np.sum(Y * np.log(Y / X) - Y + X)
        
        #In the case of Itakura–Saito divergence
        elif loss_func == "IS":
            #Update the basements
            H, U, X = get_nonzero(H, U)
            denom_H = np.sqrt(X**-1 @ U.T)
            H = H * np.sqrt((Y / X**2) @ U.T) / denom_H
            H = H / H.sum(axis=0, keepdims=True) #Normalization
            
            #Update the activation
            H, U, X = get_nonzero(H, U)
            denom_U = np.sqrt(H.T @ X**-1)
            U = U * np.sqrt(H.T @ (Y / X**2)) / denom_U
            
            #Compute the loss function
            H, U, X = get_nonzero(H, U)
            loss[i] = np.sum(Y / X - np.log(Y / X) - 1)
        
        else:
            print("The deviation shold be either 'EU', 'KL', or 'IS'.")
            sys.exit()

    
    return H, U, loss    

#Main
if __name__ == "__main__":
    
    #Setup
    audiolen = None        #Cropping time (second) [Default]None(=without cropping)
    frame_length = 0.04    #STFT window width (second) [Default]0.04
    frame_shift = 0.02     #STFT window shift (second) [Default]0.02
    num_iter = 200         #The number of iteration 
    num_base = 10          #The number of basements 
    spec_type = "amp"      #The type of spectrum (amp: amplitude, pow: power) [Default]amp
    loss_func = "KL"       #Select either EU, KL, or IS divergence [Default]KL
    
    file_path = sys.argv[1]

    #Compute the Mel-scale spectrogram
    data, Fs = sf.read(file_path)
    
    #Transform stereo into monoral
    if data.ndim == 2:
        wavdata = 0.5*data[:, 0] + 0.5*data[:, 1]
    else:
        wavdata = data
    """
    #Down sample and normalize the wave
    wavdata = sg.resample_poly(wavdata, 8000, Fs)
    Fs = 8000
    
    
    #Original sound
    #data, samplerate = sf.read(wavdata)
    sf.write('Original_sound.wav', data=wavdata, samplerate=Fs)
    """
    """
    print("Original sound")
    ipd.display(ipd.Audio(data=wavdata, rate=Fs))
    """

    #Calculate the index of window size and overlap
    FL = round(frame_length * Fs)
    FS = round(frame_shift * Fs)
    OL = FL - FS
    
    #Execute STFT
    freqs, times, dft = sg.stft(wavdata, fs=Fs, window='hamm', nperseg=FL, noverlap=OL)
    arg = np.angle(dft) #Preserve the phase
    Adft = np.abs(dft) #Preserve the absolute amplitude
    if spec_type == "amp":
        Y = Adft
    elif spec_type == "pow":
        Y = Adft**2
    else:
        print("The spec_type must be either 'amp' or 'pow'.")
        sys.exit()
    
    #Display the size of input
    print("Spectrogram size (freq, time) = " + str(Y.shape))
    print("Basements for NMF = " + str(num_base) + "\n")
    
    #Call my function for updating NMF basements and activation
    H, U, loss = get_NMF(Y, num_iter, num_base, loss_func)
    print(loss)
    
    #clustering U
    #stack_data = transform_vector(time_series_array=U)
    seed = 0
    np.random.seed(seed)
    #stack_data = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(U)
    stack_data = zscore(U, axis=1)
    print(stack_data.shape)
    #alg: kmeans or kshape
    cl = Clustering(k_init=2, n_init=10, verbose=False, random_state=seed, ic="bic", alg="kshape")
    y = cl.fit(stack_data)
    print(y.labels_)
    
    numcl = np.amax(y.labels_) + 1


    plt.figure(figsize=(16,9))
    #Pick up each basement vector
    for j in range(num_base):
        zero_mat = np.zeros((H.shape[1], H.shape[1]))
        zero_mat[j, j] = 1
        Base0 = (H @ zero_mat) @ U #Extract an only dictionary
        #print(Base0.shape)
        Base = Base0 * np.exp(1j*arg) #Restrive the phase from original wave
        #print(Base)

        _, Base_wav = sg.istft(Base, fs=Fs, window='hamm', nperseg=FL, noverlap=OL)
        plt.subplot(num_base, 1, 1 + j)
        plt.plot(Base, "k-", alpha = .2)
        plt.title("Base %d" % (j + 1))
        #data, samplerate = sf.read(Base_wav)
        sf.write('Base{}.wav'.format(j), data=Base_wav, samplerate=Fs)
        if j ==0:
            Sum_wav = Base_wav
        else:
            Sum_wav = Sum_wav + Base_wav
    #data, samplerate = sf.read(Sum_wav)
    plt.tight_layout()
    plt.show()

    sf.write('Sum.wav', data=Sum_wav, samplerate=Fs)       

    #クラスタリングして可視化(時間変化情報)
    plt.figure(figsize=(16,9))
    for yi in range(numcl):
     plt.subplot(numcl, 1, 1 + yi)
     for xx in stack_data[y.labels_ == yi]:
      plt.plot(xx.ravel(), "k-", alpha=.2)
     #plt.plot(ks.cluster_centers_[yi].ravel(), "r-")
     plt.title("Cluster %d" % (yi + 1))

    plt.tight_layout()
    plt.show()
     
    
    plt.figure(figsize=(16,9))
    for i in range(numcl):
      cluster_wav = np.zeros(Sum_wav.shape[0])  
      cluster = np.zeros((H.shape[0],U.shape[1]))
      print(cluster.shape)
      for j in range(num_base):  
        if y.labels_[j] == i:
         zero_mat = np.zeros((H.shape[1], H.shape[1]))    
         zero_mat[j, j] = 1
         Base = (H @ zero_mat) @ U
         cluster += Base
         Base = Base * np.exp(1j*arg)
         
         _, Base_wav = sg.istft(Base, fs=Fs, window='hamm', nperseg=FL, noverlap=OL)
         cluster_wav = cluster_wav + Base_wav
      plt.subplot(numcl, 1, 1 + i)
      plt.plot(cluster_wav, "k-", alpha=.8)
      plt.title("Cluster %d" % (i + 1))
      sf.write('cluster{}.wav'.format(i+1), data=cluster_wav, samplerate=Fs) 
    plt.tight_layout()
    plt.show()
      
    