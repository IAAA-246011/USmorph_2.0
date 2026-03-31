import PIL.Image as I
import numpy as np
import glob,os
from tqdm import tqdm

PI=3.141592653
def get_pos(arr):
    x,y=arr.shape
    #print(arr.shape)
    center_x,center_y=int(x/2),int(y/2)
    max_=np.max(arr)
    #print(max_)
    min_=np.min(arr)
    index=np.where(arr==max_)
    #print(index)
    x_lst,y_lst=index
    ret=[[(x-center_x)**2+(y-center_y)**2,x,y]for x,y in zip(x_lst,y_lst)]
    ret.sort()
    max_x,max_y=ret[0][1:]
    index=np.where(arr==min_)
    #print(index)
    x_lst,y_lst=index
    ret=[[(x-center_x)**2+(y-center_y)**2,x,y]for x,y in zip(x_lst,y_lst)]
    #print(ret)
    ret.sort()
    #print(ret,'===ret====')
    #print(ret[0][1:])
    min_x,min_y=ret[0][1:]

    return max_x,max_y,min_x,min_y

def give_angle(x,y):
    try:
        T=np.arctan(abs(x/y))
    except:
        if x >0:
            return PI/2
        else:
            return 3/2*PI
    if x>=0 and y >=0:
        return T
    if x >=0 and y<=0:
        return PI-T
    if x<=0 and y>=0:
        return -T
    if x<=0 and y<=0:
        return PI+T
        
def mirror(arr):
    x,y=arr.shape
    ret=np.zeros([x,2*y])
    for yy in range(y):
        ret[:,yy]=arr[:,-yy]
        ret[:,-yy]=arr[:,-yy]
    return ret

def save_arr(arr,name):
    img=I.fromarray(np.uint8(arr))
    img.save("result//"+name)
    
def show_arr(arr):
    img=I.fromarray(np.uint8(arr))
    img.show()
    
def polar(arr,N=10,angle_inter=0.05):
    max_x,max_y,min_x,min_y=get_pos(arr)
    # print(arr.shape)
    # print(max_x,max_y)
    # print(arr[max_x][max_x],'==',arr[min_x][min_x])
    # print(min_x,min_y)
    # exit()
    img=I.fromarray(np.uint8(arr))
    raw_size=arr.shape

    raw_polar_size=[int(raw_size[0]/2)*2,int(2*raw_size[0]*PI)]
    new_size=[raw_size[0]*N,raw_size[1]*N]

    X=int(new_size[0]/2)

    Y=int(2 * PI/angle_inter)
    # print(Y)
    img_resize=img.resize(new_size,resample=I.NEAREST)
    img_resize_arr=np.array(img_resize)
    # print(max_x,max_y,min_x,min_y)
    # print('-------------')
    max_x,max_y,min_x,min_y=max_x*N+N-1,max_y*N+N-1,min_x*N+N-1,min_y*N+N-1
    # print(max_x,max_y,min_x,min_y)
    # print('-------------')
    base_angle=[min_x-max_x,min_y-max_y]
    # print(base_angle)
    # print('=============')
    T=give_angle(base_angle[0],base_angle[1])
    polar_img = np.zeros((Y,X), dtype=np.uint8)

    # print(Y)
    # exit()
    for y in range(Y):
        the_angle=y*angle_inter
        # print(the_angle)
        #exit()
        for x in range(X):
            that_y=int(x*np.cos(T+the_angle)+max_y)
            that_x=int(x*np.sin(T+the_angle)+max_x)
            # print(that_x,that_y,'-=========--------======')
            # print(y,x,'---')
            try:
                polar_img[y,x]=img_resize_arr[that_x,that_y]
            except Exception as e:
                continue
    
    polar_img_m=mirror(polar_img)
    polar_img_m_=I.fromarray(polar_img_m)    
    #polar_img_=I.fromarray(polar_img)
    #polar_img_=polar_img_.resize((14,175),resample=I.NEAREST)
    # polar_img_m_1=polar_img_m_.resize(raw_polar_size,resample=I.NEAREST)
    return np.array(polar_img_m_)

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

    
def piliangzhankai(path,fi,name,angle=0,n=2):
    img=I.open(path).convert("L")
    img_rotate_r=np.array(img)
    p_1=polar(img_rotate_r,N=n)
    image = I.fromarray(p_1).convert('L')
    image = image.save(make_dir('polar_img/') + fi)
    return image
    
if __name__=="__main__":
    # pathnames = ['train','test']
    pathnames = ['cae_img']
    #如果要展开的代码无train和test文件pathnames可删去    
    k = 0
    for pathname in (pathnames):
        filepath=os.listdir('%s/'%pathname)
        #这里面是原文件的地址
        for f in filepath:
            img = piliangzhankai('%s/'%pathname+f,f,pathname)
            #()中为原文件地址，文件名，pathnames根据需要设定，如果不需要删去即可，但同时piliangzhankai()中pathname相关也记得删去
            k = k + 1
            print(f,'====%s====='%k)

          
