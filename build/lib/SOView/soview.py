import numpy as np
import time
from skimage.color import rgb2lab, lab2rgb
from sklearn.preprocessing import *
from sklearn.decomposition import PCA
from scipy.sparse import issparse 
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import anndata as ad
import scanpy.external as sce


import sklearn.datasets
import umap
# import umap.plot

import scanpy as sc
import squidpy as sq

sc.logging.print_header()
print(f"squidpy=={sq.__version__}")
import seaborn as sns

import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.multioutput import MultiOutputRegressor


sns.set_style('white')
# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 2}

# matplotlib.rc('font', **font)
# matplotlib.rcParams.update({'font.size': 2})

import matplotlib.pyplot as plt

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title




import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.multioutput import MultiOutputRegressor


def SOViewer_uniformUmapRF(
    adata,
    pca=100,
    sample_rate=0.01,
    dot_size=0.01,
    densmap=False,
    CT_obs='cluster',
    max_depth = 100,
    n_estimators=100,
    marker='o',
    plot_mode='image',
    save=None
):
    print('#########################################')
    print('sketching data of size {0}...'.format(adata.X.shape))
    time_start_whole = time.time()
    time_start=time.time()
#     th = treehopper(adata.X, partition=PCATreePartition, max_partition_size=sketch_max_partition_size)
#     th.hop(sketch_size)
#     adata_sub = adata[th.path]
    adata_sub = sc.pp.subsample(adata,sample_rate,copy=True)
    time_end=time.time()
    print('sketching time cost',time_end-time_start,'s')


    print('#########################################')
    print('umap on sketch data of size {0}...'.format(adata_sub.shape))
    time_start=time.time()
    sc.pp.pca(adata_sub,n_comps=pca)
    sc.pp.neighbors(adata_sub)
    sc.tl.umap(adata_sub,n_components=3)
    time_end=time.time()
    print('umap time cost',time_end-time_start,'s')

    train_x = adata_sub.X
    train_y = adata_sub.obsm['X_umap']
    test_x = adata.X
    if issparse(test_x):
        test_x = test_x.toarray()

    print('#########################################')
    print('RF training on sketch data of size {0}...'.format(train_x.shape))
    time_start=time.time()
    clf = MultiOutputRegressor(RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators, random_state=0,n_jobs=-1),n_jobs=-1).fit(train_x, train_y)
    time_end=time.time()
    print('RF training time cost',time_end-time_start,'s')

    print('#########################################')
    print('RF testing on sketch data of size {0}...'.format(test_x.shape))
    time_start=time.time()
    test_y = clf.predict(test_x)
    time_end=time.time()
    print('RF testing time cost',time_end-time_start,'s')

    adata.obsm['X_uniformUmap'] = test_y
    if save is not None:
        save = save+'_uniformUmap'
        
    if plot_mode=='scatter':
        SOViewer_plot(adata,embedding_use='X_uniformUmap',dot_size=dot_size,marker=marker,save=save)
    elif plot_mode=='image':
        SOViewer_plot_image(adata,embedding_use='X_uniformUmap',save=save)
    else:
        print('no plot')
        return

    
    

def SOViewer_geosketchUmapRF(
    adata,pca=100,
    sample_rate=0.01,
    sketch_max_partition_size=500,
    dot_size=0.01,
    densmap=False,
    CT_obs='cluster',
    max_depth = 100,
    n_estimators=100,
    marker='o',
    plot_mode='image',
    save=None
):




    sketch_size = int(sample_rate*adata.shape[0])
    print('#########################################')
    print('sketching data of size {0}...'.format(adata.X.shape))
    time_start_whole = time.time()
    time_start=time.time()
    if issparse(adata.X):
        to_sketch = adata.X.toarray()
    else:
        to_sketch = adata.X
        
    # th = treehopper(to_sketch, partition=PCATreePartition, max_partition_size=sketch_max_partition_size)
    # th.hop(sketch_size)
    from geosketch import gs
    import fbpca 
    U, s, Vt = fbpca.pca(to_sketch, k=pca) # E.g., 100 PCs.
    X_dimred = U[:, :pca] * s[:pca]
    sketch_index = gs(X_dimred, sketch_size, replace=False)

    # X_sketch = X_dimred[sketch_index]
    adata_sub = adata[sketch_index]
    time_end=time.time()
    print('sketching time cost',time_end-time_start,'s')


    print('#########################################')
    print('umap on sketch data of size {0}...'.format(adata_sub.shape))
    time_start=time.time()
    sc.pp.pca(adata_sub,n_comps=pca)
    sc.pp.neighbors(adata_sub)
    sc.tl.umap(adata_sub,n_components=3)
    time_end=time.time()
    print('umap time cost',time_end-time_start,'s')

    train_x = adata_sub.X
    train_y = adata_sub.obsm['X_umap']

    test_x = adata.X
    if issparse(test_x):
        test_x = test_x.toarray()

    print('#########################################')
    print('RF training on sketch data of size {0}...'.format(train_x.shape))
    time_start=time.time()
    clf = MultiOutputRegressor(RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators, random_state=0,n_jobs=-1),n_jobs=-1).fit(train_x, train_y)
    time_end=time.time()
    print('RF training time cost',time_end-time_start,'s')

    print('#########################################')
    print('RF testing on sketch data of size {0}...'.format(test_x.shape))
    time_start=time.time()
    test_y = clf.predict(test_x)
    time_end=time.time()
    print('RF testing time cost',time_end-time_start,'s')

    adata.obsm['X_sketchUmap'] = test_y
    if save is not None:
        save = save+'_sketchUmap'
        
    if plot_mode=='scatter':
        SOViewer_plot(adata,embedding_use='X_sketchUmap',dot_size=dot_size,marker=marker,save=save)
    elif plot_mode=='image':
        SOViewer_plot_image(adata,embedding_use='X_sketchUmap',save=save)
    else:
        print('no plot')
        return
    
    
def SOViewer_PCA(adata,dot_size=0.001,marker='o',plot_mode='image',save=None):
    # 输入：
    # adata(附带spatial, 处理过后的X)
    # sample_rate：默认0.01


    print('projecting all data into PCA space...')
    time_start=time.time()
    sc.pp.pca(adata,n_comps=3)
    time_end=time.time()
    print('projecting time cost',time_end-time_start,'s')
    if save is not None:
        save = save+'_PCA'
        
    if plot_mode=='scatter':
        SOViewer_plot(adata,embedding_use='X_pca',dot_size=dot_size,marker=marker,save=save)
    elif plot_mode=='image':
        SOViewer_plot_image(adata,embedding_use='X_pca',save=save)
    else:
        print('no plot')
        return


    
    
def SOViewer_UMAP(adata,pca=100,dot_size=0.001,marker='o',plot_mode='image',save=None):
    # 输入：
    # adata(附带spatial, 处理过后的X)
    # sample_rate：默认0.01


    print('projecting all data into PCA space...')
    time_start=time.time()
    
    sc.pp.pca(adata,n_comps=pca)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata,n_components=3)
#     test_embedding = adata.obsm['X_umap']
    time_end=time.time()
    print('projecting time cost',time_end-time_start,'s')

    if save is not None:
        save = save+'_UMAP'
        
    if plot_mode=='scatter':
        SOViewer_plot(adata,embedding_use='X_umap',dot_size=dot_size,marker=marker,save=save)
    elif plot_mode=='image':
        SOViewer_plot_image(adata,embedding_use='X_umap',save=save)
    else:
        print('no plot')
        return
    

def SOViewer_plot(adata,save=None,embedding_use='X_umap',dot_size=0.001,marker='o',plot_cluster=None):
# save is the full file name without extension, e.g. pdf
    print('generating color coding...')
    test_embedding = np.array(adata.obsm[embedding_use])
    rgb = MinMaxScaler(clip=True).fit_transform(test_embedding)
    lab = np.zeros_like(rgb)
    lab[:,0] = MinMaxScaler(feature_range=(0, 100),clip=True).fit_transform(rgb[:,0][:,None])[:,0]
    lab[:,1] = MinMaxScaler(feature_range=(-128, 127),clip=True).fit_transform(rgb[:,1][:,None])[:,0]
    lab[:,2] = MinMaxScaler(feature_range=(-128, 127),clip=True).fit_transform(rgb[:,2][:,None])[:,0]
    lab = lab2rgb(lab)

    c = lab
    print(c.max(),c.min())
    plt.scatter(x=adata.obsm['spatial'][:,0],y=adata.obsm['spatial'][:,1],c=c,s=dot_size,edgecolors='none',marker=marker)
    plt.gca().set_aspect('equal', adjustable='box')
#     plt.title('lab')
    plt.title('LAB, {0} spots, {1} features'.format(adata.shape[0],adata.shape[1]))
    

#     plt.show()
    if save is None:
        plt.show()
    else:
        plt.savefig('{0}_LAB.pdf'.format(save),format='pdf',bbox_inches='tight')
        
        

    c = rgb
    print(c.max(),c.min())
    
    plt.scatter(x=adata.obsm['spatial'][:,0],y=adata.obsm['spatial'][:,1],c=c,s=dot_size,edgecolors='none',marker=marker)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('RGB, {0} spots, {1} features'.format(adata.shape[0],adata.shape[1]))
    if save is None:
        plt.show()
    else:
        plt.savefig('{0}_RGB.pdf'.format(save),format='pdf',bbox_inches='tight')
    
    if plot_cluster is None:
        return
    c = np.array(adata.obs[plot_cluster]).astype('int')
    
    plt.scatter(x=adata.obsm['spatial'][:,0],y=adata.obsm['spatial'][:,1],c=c,s=dot_size,edgecolors='none',marker=marker)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('cluster on spatial')
    plt.show()
        

        
def SOViewer_plot_image(adata,save=None,embedding_use='X_umap'):
# save is the full file name without extension, e.g. pdf
    print('generating color coding...')
    test_embedding = np.array(adata.obsm[embedding_use])
    rgb = MinMaxScaler(clip=True).fit_transform(test_embedding)
    lab = np.zeros_like(rgb)
    lab[:,0] = MinMaxScaler(feature_range=(0, 100),clip=True).fit_transform(rgb[:,0][:,None])[:,0]
    lab[:,1] = MinMaxScaler(feature_range=(-128, 127),clip=True).fit_transform(rgb[:,1][:,None])[:,0]
    lab[:,2] = MinMaxScaler(feature_range=(-128, 127),clip=True).fit_transform(rgb[:,2][:,None])[:,0]
    lab = lab2rgb(lab)

    spatial_mat = np.array(adata.obsm['spatial'])

    c = lab
    img = scatter2image(spatial_mat,c)
    plt.imshow(img)
    plt.title('LAB, {0}x{1} image, {2} features'.format(img.shape[0],img.shape[1],adata.shape[1]))
    plt.axis('off')
    if save is None:
        plt.show()
    else:
        plt.savefig('{0}_LAB.pdf'.format(save),format='pdf',bbox_inches='tight')
        
        

    c = rgb
    img = scatter2image(spatial_mat,c)
    plt.imshow(img)
    plt.title('RGB, {0}x{1} image, {2} features'.format(img.shape[0],img.shape[1],adata.shape[1]))
    plt.axis('off')
    if save is None:
        plt.show()
    else:
        plt.savefig('{0}_RGB.pdf'.format(save),format='pdf',bbox_inches='tight')
        


        
        
# 自动把scatter转化为image,最大适应
def scatter2image(spatial_mat,c): 
    spatial_mat = spatial_mat.astype('int')
    x_offset = spatial_mat[:,0].min()
    y_offset = spatial_mat[:,1].min()
    x_sz = int(spatial_mat[:,0].max()-spatial_mat[:,0].min()+1)
    y_sz = int(spatial_mat[:,1].max()-spatial_mat[:,1].min()+1)
    adjusted_mat = np.zeros(shape=(x_sz,y_sz,3))
    count_mat = np.zeros(shape=(x_sz,y_sz))
    for i in range(spatial_mat.shape[0]):
        cur_spatial = spatial_mat[i]
        cur_triple_new = c[i]
        cur_count_old = count_mat[int(cur_spatial[0]-x_offset),int(cur_spatial[1]-y_offset)]
        cur_triple_old = adjusted_mat[int(cur_spatial[0]-x_offset),int(cur_spatial[1]-y_offset),:]
        cur_count_update = cur_count_old+1
        cur_triple_update = (cur_triple_old*cur_count_old+cur_triple_new)/cur_count_update
        
        count_mat[int(cur_spatial[0]-x_offset),int(cur_spatial[1]-y_offset)]=cur_count_update
        adjusted_mat[int(cur_spatial[0]-x_offset),int(cur_spatial[1]-y_offset),:] = cur_triple_update
    return adjusted_mat
