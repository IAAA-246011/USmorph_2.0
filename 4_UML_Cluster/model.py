#!-*-coding:utf-8-*-
from sklearn.cluster import KMeans,AgglomerativeClustering,AffinityPropagation,MeanShift,estimate_bandwidth,SpectralClustering,DBSCAN,Birch
from sklearn.datasets.samples_generator import make_blobs
from sklearn.mixture import GaussianMixture
import glob
import shutil
import os
import time
class model(object):
    def __init__(self,class_num):
        self.class_num=class_num
        #self.build(data)
    def build(self,model_type):
        self.model_type=model_type
        if self.model_type=="kmeans":
            self.estimator = KMeans(n_clusters=self.class_num)
            print("选用算法k_means")
        elif self.model_type=="agg":
            self.estimator = AgglomerativeClustering(n_clusters=self.class_num, linkage='ward')
            print("选用算法 Agg")
        elif self.model_type=="ap":
            print("选用算法 ap")
            self.estimator= AffinityPropagation()#preference=-50#自动寻找聚类类别个数
            #self.class_num=len(self.estimator.cluster_centers_indices)
        elif self.model_type=="mean-shift":
            print("选用算法mean-shift")
            #bandwidth = estimate_bandwidth(data, quantile = 0.2, n_samples = 500)
            self.estimator=MeanShift()#bandwidth = bandwidth, bin_seeding = True#自动寻找聚类类别个数
        elif self.model_type=="spectral":
            print("选用算法spectral")
            self.estimator=SpectralClustering(n_clusters=self.class_num)#,affinity='precomputed',二维才适用
        elif self.model_type=="dbscan":
            print("选用算法dbscan")
            self.estimator=DBSCAN()#eps=3, min_samples=2,自动寻找聚类类别个数
        elif self.model_type=="gmm":#Gaussian Mixture Model not true clustering
            print("选用算法gmm")
            self.estimator=GaussianMixture(n_components=self.class_num)
        elif self.model_type=="birch":
            print("选用算法birch")
            self.estimator=Birch(n_clusters =self.class_num)
        else:
            print("wrong model type, please check config")
            exit()
    def run(self,data):
        print("正在聚类")
        st=time.time()
        self.estimator.fit(data)
        print("聚类完成，用时 %s 秒"%(time.time()-st))
    def show(self,file_list):
        self.label_pred = self.estimator.labels_
        if glob.glob("result/%s"%self.model_type)==[]:
            os.system("mkdir result/%s"%self.model_type)
        for label in set(self.label_pred):
            if glob.glob("result/%s/%s"%(self.model_type,label))!=[]:
                os.system("rm -r result/%s/%s"%(self.model_type,label))
            os.mkdir("result/%s/%s"%(self.model_type,label))
        for file_path,label in zip(file_list,self.label_pred):
            os.system("cp %s result/%s/%s"%(file_path,self.model_type,label))  
        print(self.label_pred)
    def show_w(self,file_list):
        self.label_pred = self.estimator.labels_
        if glob.glob("result\\%s"%self.model_type)==[]:
            os.system("mkdir result\\%s"%self.model_type)
        for label in set(self.label_pred):
            if glob.glob("result\\%s\\%s"%(self.model_type,label))!=[]:
                os.system("rm -r result\\%s\\%s"%(self.model_type,label))
            os.mkdir("result\\%s\\%s"%(self.model_type,label))
        for file_path,label in zip(file_list,self.label_pred):
            os.system("cp %s result\\%s\\%s"%(file_path,self.model_type,label))  
        print(self.label_pred)
        
        
