import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize



def fun_CU(theta, data, k, cl, qq): #cl must be an np.array
    VV = 0
    p = data.shape[1] 
    for i in range(k):
        if np.sum(cl==i)>0:
            nn = np.sum(cl==i)
            xx = data.values[cl==i]
            a  = np.sum((theta+((1-2*theta)*(xx < np.tile(qq[i,], (nn,1))))) * #compute the distance of the entrie dataset wrt each cluster
        np.absolute(xx - np.tile(qq[i,], (nn,1))),axis=1)
            VV =+ np.sum(a)
    n = len(cl)
    VV = VV - p * n * np.log(theta*(1-theta))
    return VV

#the clustering algorithm
def alg_CU(data, k = 2,  eps = 1e-8, it_max = 100, B = 30): 
    #k default number of clusters,it_max is the maximum number of iterations of the algorithms, B is the number of different initializations of the clustering
    
    #initialization, and extraction of the number of observations and variables
    nobs = data.shape[0]
    p = data.shape[1]
    #creating zeros matrix that we will use later 
    qq = np.zeros((k,p))
    QQ = np.zeros((nobs,k))
    VV_temp = None 
    VV_list = np.array([math.inf])

   
    for j in range(B):
        theta = np.random.uniform(0,1)
        for i in range(k):
            if k == 1:
                qq[i,] = np.apply_along_axis(np.quantile,0,data, theta/2) 
            else : #calculate the quantile of the variable wrt to each cluster
                qq[i,] = np.apply_along_axis(np.quantile,0,data,((i+1) - 1)/(k - 1) * 0.5 + theta/2) 

            #then calculate the quantile of the variable wrt to each cluster but each cluster now has slightly different theta (only in this step 
            # we will modify theta)
            QQ[:,i] = np.sum((theta+((1-2*theta)*(data.values < np.tile(qq[i,], (nobs,1))))) *
                np.absolute(data.values-np.tile(qq[i,], (nobs,1))),axis=1) - p*np.log(theta*(1-theta))

        #extract the cluster assigment, by assign the observation to the cluster to which it is nearest      
        cl = np.apply_along_axis(np.argmin,1,QQ)
        #calulate within cluster sum of distances
        VV_temp = np.sum(QQ[range(nobs), cl])
        if VV_temp < VV_list : #check that the new likelihood distance is  less
            VV_list = VV_temp       #than the lowest one until now and  update partition and parameter theta
            cl_true = cl
            theta_true = theta

    cl = cl_true
    theta = theta_true
    ratio = 5
    h = 0
    while ratio > eps and h < it_max :
        h = h + 1
        #calculate cluster size:
        #at each step inizialize an empty dictionary with k cluster
        nk = dict(zip(range(k),np.repeat(0, k, axis=0)))
        #add the observation to each cluster to which it is assign
        values, counts = np.unique(cl, axis=0, return_counts=True)
        #we do all this because otherwise we may obtain an cluster with 0 observation we still keep the order 

        #create the dictionary in which we will save all cluster assigments, the partitions
        nk_new = dict(zip(values,counts))
        nk.update(nk_new)

        #update the clusters parameters and related values
        for i in range(k):
            if nk[i] > 0 :
                qq[i,] = np.apply_along_axis(np.quantile,0,data.values[cl==i],theta) #find new cluster quantiles
                
                #compute the distance of the entrie dataset wrt each cluster
                QQ[:,i] = np.sum((theta+((1-2*theta)*(data.values < np.tile(qq[i,], (nobs,1))))) * 
                np.absolute(data.values-np.tile(qq[i,], (nobs,1))),axis=1) - p*np.log(theta*(1-theta))
        
        #Now update theta by finding the one that minimize the function above
        cl = np.apply_along_axis(np.argmin,1,QQ) #assign each observation to the cluster with closest quantile
        #update theta
        theta = minimize(fun_CU, x0=theta,args = (data, k, cl, qq), bounds=((1e-4,0.99),)).x[0]

        VV_list = np.append(VV_list, np.sum(QQ[range(nobs), cl]))
        ratio = (VV_list[h-1,] - VV_list[h,]) / VV_list[h-1,]
        if h < 5: #in order to make at leat 5 iterations
            ratio = 2 * eps

    #save the quantile in a dataframe        
    qq1 = pd.DataFrame(qq)
    qq1.columns = data.columns
    res = { 
        'Vseq' : VV_list, #the entire list of sum of distances (likelihood of the model at each iterations)
        'VV' : VV_list[h,], #last sum of distances
        'cl' : cl, #last clsuter assigment
        'qq' : qq1, #last matrix with all distances
        'theta' : theta}  #the quantile theta

    return res



def fun_CS(theta, data, k, cl, qq, lam): #cl must be an np.array
    VV = 0
    p = data.shape[1] 
    for i in range(k):
        if np.sum(cl==i)>0:
            nn = np.sum(cl==i)
            xx = data.values[cl==i]
            a = np.sum(np.tile(lam,(nn,1))*(theta+((1-2*theta)*(xx < np.tile(qq[i,], (nn,1))))) * np.absolute(xx-np.tile(qq[i,], (nn,1))),axis=1)
            VV =+ np.sum(a) - np.sum(nn*np.log(lam*theta * (1-theta)))
    n = len(cl)
    VV = VV - p * n * np.log(theta*(1-theta))
    return VV



def alg_CS(data, k = 2,  eps = 1e-8, it_max = 100, B = 30): 
    
    
    nobs = data.shape[0]
    p = data.shape[1]
    lam = np.ones(p)
    qq = np.zeros((k,p))
    QQ = np.zeros((nobs,k))
    QQ0 =  np.zeros((k, nobs, p))
    VV_temp = None 
    VV_list = np.array([math.inf])   

    #initialization is the same as the one for common theta and unscaled variables
    for j in range(B):
        theta = np.random.uniform(0,1)
        for i in range(k):
                if k == 1:
                    qq[i,] = np.apply_along_axis(np.quantile,0,data, theta/2) 
                else :
                    qq[i,] = np.apply_along_axis(np.quantile,0,data,((i+1) - 1)/(k - 1) * 0.5 + theta/2) 

            
                QQ[:,i] = np.sum((theta+((1-2*theta)*(data.values < np.tile(qq[i,], (nobs,1))))) *   np.absolute(data.values-np.tile(qq[i,], (nobs,1))),axis=1) - p*np.log(theta*(1-theta))

             
        cl = np.apply_along_axis(np.argmin,1,QQ)
        VV_temp = np.sum(QQ[range(nobs), cl])
        if VV_temp < VV_list : 
            VV_list = VV_temp       
            cl_true = cl
            theta_true = theta
    # CLUSTERING
    cl = cl_true
    theta = theta_true
    ratio = 5
    h = 0
    #we will iterate untill  the following condition holds
    while ratio > eps and h < it_max :
        h = h + 1
        nk = dict(zip(range(k),np.repeat(0, k, axis=0)))
        values, counts = np.unique(cl, axis=0, return_counts=True)
        nk_new = dict(zip(values,counts))
        nk.update(nk_new)
        
        #compute the new barycenters
        for i in range(k):
            if nk[i] > 0 : 
                qq[i,] = np.apply_along_axis(np.quantile,0,data.values[cl==i],theta) #find new cluster quantiles
                QQ0[[i]] = (theta+((1-2*theta)*(data.values < np.tile(qq[i,], (nobs,1))))) * np.absolute(data.values-np.tile(qq[i,], (nobs,1)))
                #compute the distance of the entrie dataset wrt each cluster
                QQ[:,i] = np.sum((np.tile(lam,(nobs,1))*QQ0[[i]])[0],axis=1) - np.sum(np.log(lam*theta * (1-theta)))

        #compute theta        
        theta = minimize(fun_CS, x0=theta,args = (data, k, cl, qq,lam), bounds=((1e-4,0.99),)).x[0]

        #compute lambda
        sel = np.zeros((nobs,p))
        for i in range(nobs):
            sel[i] = QQ0[cl[i],i]
        dem = np.average(sel,axis=0)
        dem = np.where(dem == 0,eps,dem)
        lam = 1/dem
                #compute lambda

        
        cl = np.apply_along_axis(np.argmin,1,QQ) 
        VV_list = np.append(VV_list, np.sum(QQ[range(nobs), cl]))
        ratio = (VV_list[h-1,] - VV_list[h,]) / VV_list[h-1,]
        if h < 5: #in order to make at leat 5 iterations
            ratio = 2 * eps
         
    qq1 = pd.DataFrame(qq)
    qq1.columns = data.columns
    res = lt ={ 
        'Vseq' : VV_list, #the entire list of sum of distances (likelihood of the model at each iterations)
        'VV' : VV_list[h,], #last sum of distances
        'cl' : cl, #last clsuter assigment
        'qq' : qq1, #last matrix with all distances
        'theta' : theta,
        'lambda' : lam}  #the quantile theta

    return res        

def fun_VU(theta, data, k, cl, qq): #cl must be an np.array
    VV = 0
    for i in range(k):
        if np.sum(cl==i)>0:
            nn = np.sum(cl==i)
            xx = data.values[cl==i]
            VV =+ np.sum((theta+((1-2*theta)*(xx < np.tile(qq[i], (1,nn))))) * np.absolute(xx - np.tile(qq[i], (1,nn)))) #compute the distance of the entrie dataset wrt each cluster
            
    n = len(cl)
    VV = VV -  n * np.log(theta*(1-theta))
    return VV



def alg_VU(data, k = 2,  eps = 1e-8, it_max = 100, B = 30): 
    
    nobs = data.shape[0]
    p = data.shape[1]
    qq = np.zeros((k,p))
    QQ = np.zeros((p,nobs,k))
    VV_temp = None 
    VV_list = np.array([math.inf])

    
    for hh in range(B):
        theta = np.random.uniform(0,1,p)
        for j in range(p):
            for i in range(k):
                if k == 1:
                            qq[i,j] = np.quantile(data.values[:,j],theta[j]/2) 
                else : #calculate the quantile of the variable wrt to each cluster
                            qq[i,j] = np.quantile(data.values[:,j],((i+1) - 1)/(k - 1) * 0.5 + theta[j]/2) 

                #then calculate the quantile of the variable wrt to each cluster but each cluster now has slightly different theta (only in this step 
                # we will modify theta)
                QQ[j,:,i] = (theta[j]+((1-2*theta[j])*(data.values[:,j] < np.tile(qq[i,j], (1,nobs))))) * np.absolute(data.values[:,j]-np.tile(qq[i,j], (1,nobs))) - np.log(theta[j]*(1-theta[j]))

           
        cl = np.apply_along_axis(np.argmin,1,np.sum(QQ,0))
        VV_temp= 0
    
        for j in range(p):
            VV_temp =+ np.sum(QQ[j,range(nobs), cl])

        if VV_temp < VV_list : 
            VV_list = VV_temp       
            cl_true = cl
            theta_true = theta
        #at the end of the B=30 different initialization keep the clustering partition that yield the smallest VV


    cl = cl_true
    theta = theta_true
    ratio = 5
    h = 0
    while ratio > eps and h < it_max :
        h = h + 1
        
        nk = dict(zip(range(k),np.repeat(0, k, axis=0)))
        values, counts = np.unique(cl, axis=0, return_counts=True)
        nk_new = dict(zip(values,counts))
        nk.update(nk_new)

        #update the clusters parameters and related values
        for j in range(p):
            for i in range(k):
                if nk[i] > 0 : #for each clsuter non empty
                    qq[i,j] = np.quantile(data.values[cl==i,j],theta[j]) #find new cluster quantiles
                    #compute the distance of the entrie dataset wrt each cluster, no row sum this time, we will sum element of the different matrices in the tensor
                    QQ[j,:,i] = (theta[j]+((1-2*theta[j])*(data.values[:,j] < np.tile(qq[i,j], (1,nobs))))) * np.absolute(data.values[:,j]-np.tile(qq[i,j], (1,nobs))) - np.log(theta[j]*(1-theta[j]))

        
     
        cl = np.apply_along_axis(np.argmin,1,np.sum(QQ,0))
        #update theta
        for j in range(p):
            theta[j] = minimize(fun_VU, x0=theta[j],args = (data.iloc[:,j], k, cl, qq[:,j]), bounds=((1e-4,0.99),)).x[0]
        

        VV_temp= 0
        #calculate the within cluster sum of distance
        for j in range(p):
            VV_temp =+ np.sum(QQ[j,range(nobs), cl])
        VV_list = np.append(VV_list, VV_temp)
        ratio = (VV_list[h-1,] - VV_list[h,]) / VV_list[h-1,]
        if h < 5: 
            ratio = 2 * eps

          
    qq1 = pd.DataFrame(qq)
    qq1.columns = data.columns
    res = { 'Vseq' : VV_list, #the entire list of sum of distances (likelihood of the model at each iterations)
        'VV' : VV_list[h,], #last sum of distances
        'cl' : cl, #last clsuter assigment
        'qq' : qq1, #last matrix with all distances
        'theta' : theta}  #the quantile theta

    return res



def fun_VS(theta, data, k, cl, qq, lam): #cl must be an np.array
    VV = 0
    for i in range(k):
        if np.sum(cl==i)>0:
            nn = np.sum(cl==i)
            xx = data.values[cl==i]
            VV =+ lam*np.sum((theta+((1-2*theta)*(xx < np.tile(qq[i,], (1,nn))))) * np.absolute(xx-np.tile(qq[i,], (1,nn))))
    n = len(cl)
    VV = VV - n * np.log(lam*theta*(1-theta))
    return VV



def alg_VS(data, k = 2,  eps = 1e-8, it_max = 100, B = 30): 
    

    nobs = data.shape[0]
    p = data.shape[1]
    lam = np.ones(p)
    qq = np.zeros((k,p))
    QQ = QQ0 =  np.zeros((p,nobs,k))
    VV_temp = None 
    VV_list = np.array([math.inf])   
    
    #initialization is the same as the one for common theta and unscaled variables
    for hh in range(B):
        theta = np.random.uniform(0,1,p)
        for j in range(p):
            for i in range(k):
                if k == 1:
                            qq[i,j] = np.quantile(data.values[:,j],theta[j]/2) 
                else : #calculate the quantile of the variable wrt to each cluster
                            qq[i,j] = np.quantile(data.values[:,j],((i+1) - 1)/(k - 1) * 0.5 + theta[j]/2) 

                #then calculate the quantile of the variable wrt to each cluster but each cluster now has slightly different theta (only in this step 
                # we will modify theta)
                QQ[j,:,i] = (theta[j]+((1-2*theta[j])*(data.values[:,j] < np.tile(qq[i,j], (1,nobs))))) * np.absolute(data.values[:,j]-np.tile(qq[i,j], (1,nobs))) - np.log(theta[j]*(1-theta[j]))

             
        cl = np.apply_along_axis(np.argmin,1,np.sum(QQ,0))
        VV_temp= 0
        
        for j in range(p):
            VV_temp =+ np.sum(QQ[j,range(nobs), cl])

        if VV_temp < VV_list :
            VV_list = VV_temp       
            cl_true = cl
            theta_true = theta

    # CLUSTERING
    cl = cl_true
    theta = theta_true
    ratio = 5
    h = 0

    while ratio > eps and h < it_max :
        h = h + 1
        #calculate cluster size nk
        nk = dict(zip(range(k),np.repeat(0, k, axis=0)))
        values, counts = np.unique(cl, axis=0, return_counts=True)
        nk_new = dict(zip(values,counts))
        nk.update(nk_new)
        
        for j in range(p):
            for i in range(k):
                if nk[i] > 0 : #for each clsuter non empty
                    qq[i,j] = np.quantile(data.values[cl==i,j],theta[j]) 
                    QQ0[j,:,i] = (theta[j]+((1-2*theta[j])*(data.values[:,j] < np.tile(qq[i,j], (1,nobs))))) * np.absolute(data.values[:,j]-np.tile(qq[i,j], (1,nobs)))
                    #compute the scale dissimilarities of the entrie dataset wrt each cluster
                    QQ[j,:,i] = lam[j] * QQ0[j,:,i] - np.log(lam[j]*theta[j]*(1-theta[j]))
                
            
        cl = np.apply_along_axis(np.argmin,1,np.sum(QQ,0)) #assign each observation to the cluster with closest quantile

        for j in range(p):
            theta[j] = minimize(fun_VS, x0=theta[j],args = (data.iloc[:,j], k, cl, qq[:,j],lam[j]), bounds=((1e-4,0.99),)).x[0]
        

        #compute lambda
        sel = np.zeros((nobs,p))
        for i in range(nobs):
            sel[i] = QQ0[:,i,cl[i]]
        dem = np.average(sel,axis=0)
        dem = np.where(dem == 0,eps,dem)
        lam = 1/dem

        
        VV_temp= 0
        #calculate the within cluster sum of distance
        for j in range(p):
            VV_temp =+ np.sum(QQ[j,range(nobs), cl])
        VV_list = np.append(VV_list, VV_temp)
        ratio = (VV_list[h-1,] - VV_list[h,]) / VV_list[h-1,]
        if h < 5: #in order to make at leat 5 iterations
            ratio = 2 * eps

    #save the quantile in a dataframe        
    qq1 = pd.DataFrame(qq)
    qq1.columns = data.columns
    res = { 'Vseq' : VV_list, #the entire list of sum of distances (likelihood of the model at each iterations)
        'VV' : VV_list[h,], #last sum of distances
        'cl' : cl, #last clsuter assigment
        'qq' : qq1, #last matrix with all distances
        'theta' : theta,
        'lambda' : lam}  

    return res    
