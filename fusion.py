from imagerie import *
from scipy.ndimage.filters import laplace
from scipy.ndimage import uniform_filter, gaussian_filter


def extract_windows(A, size, padding=True):
    # inspired by http://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python
    if padding:
        before = int((size - 1) / 2)
        after = size - 1 - before
        A = np.pad(A, ((before, after), (before, after)), 'mean')
    M,N = A.shape
    B = [M - size + 1, N - size + 1]
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1
    start_idx = np.arange(B[0])[:,None]*N + np.arange(B[1])
    offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)
    return np.take (A,start_idx.ravel()[:,None] + offset_idx.ravel())

def guided_filtering(im, guide, size=3, epsilon=0.1):
    if size % 2 != 1:
        raise ArgumentError('Size must be an odd number.')
    
    if len(guide.shape) >= 3:
        return sum([guided_filtering(im, guide[:,:,k], size, epsilon) for k in range(guide.shape[2])])
        
    # extract windows
    im_wins = extract_windows(im, size)
    guide_wins = extract_windows(guide, size)
    
    mu_k = guide_wins.mean(axis=1)
    nu_k = im_wins.mean(axis=1)
    delta_k = guide_wins.std(axis=1)
    a_k = (im_wins * guide_wins).mean(axis=1) - mu_k * nu_k / (delta_k + epsilon)
    b_k = nu_k - a_k * mu_k
    
    a_wins = extract_windows(a_k.reshape(*im.shape), size)
    b_wins = extract_windows(b_k.reshape(*im.shape), size)
    
    return a_wins.mean(axis=1).reshape(*im.shape) * guide + b_wins.mean(axis=1).reshape(*im.shape)

def extract_maps(im, average_filter_size=31, sigma_r=5):
    base_layer = uniform_filter(im, size=average_filter_size)
    detail_layer = im - base_layer
    if len(im.shape) >= 3:
        imgray = rgb2gray(im)
    else:
        imgray = im.astype(np.float64) - 0.0001
    saliency = gaussian_filter(abs(laplace(imgray)), sigma_r)
    return base_layer, detail_layer, saliency

def GFF(images,
        average_filter_size = 31,
        sigma_r=5,
        r_base=5,
        epsilon_base=2,
        r_detail=5,
        epsilon_detail=0.1,
        return_info=False):
    
    base_layers, detail_layers, saliencies = zip(*[extract_maps(im, average_filter_size, sigma_r) for im in images])
    saliency_idx = np.argmax(saliencies, axis=0)
    
    masks = [saliency_idx == i for i in range(len(images))]
    
    base_weights = [guided_filtering(mask, im / 255, size=r_base, epsilon=epsilon_base) for im, mask in zip(images, masks)]
    detail_weights = [guided_filtering(mask, im / 255, size=r_detail, epsilon=epsilon_detail) for im, mask in zip(images, masks)]
    
    # a bit hacky
    if len(images[0].shape) >= 3:
        base_weights = [w[:,:,None] for w in base_weights]
        detail_weights = [w[:,:,None] for w in detail_weights]
    
    base = sum([base * weight for base, weight in zip(base_layers, base_weights)]) / sum(base_weights)
    detail = sum([detail * weight for detail, weight in zip(detail_layers, base_weights)]) / sum(base_weights)
    
    result = base + detail
    
    if not return_info:
        return result
    
    info = {}
    info['saliencies'] = saliencies
    info['masks'] = masks
    return result, info

def cGFF(images, *args, **kwargs):
    r = GFF([i.astype(np.float64) for i in images], *args, **kwargs)
    if type(r) == tuple:
        (r[0].astype(np.uint8),) + r[1:]
        return r
    else:
        return r.astype(np.uint8)

