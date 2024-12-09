from matplotlib import pyplot as plt
from googleapiclient.discovery import build
import pprint

import cv2
import urllib
import numpy as np



my_api_key = "AIzaSyCy8CVOfXtbT20cLJrdbqwea8kqe2lF5AY"
my_cse_id = "069752570185a4edb"

def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res['items']


def imgFromUrl(url):
    raw_req = urllib.request.Request(url=url, headers={'User-Agent': 'Mozilla/6.0'})
    req = urllib.request.urlopen(raw_req)
    if req.length is None or req.length > 5000000:
        print('too long on', url)
        return None
    else:
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR) 

    
def imgsFromResults(results):
    ret = []
    for x in results:
        try:
            img = imgFromUrl(x['link'])
            if img is not None:
                ret.append(img)
        except urllib.error.HTTPError:
            print('httperror on', x['link'])
    return ret
    

def getImageResultsInner(term, num, start):
        return google_search(term, my_api_key, my_cse_id, num=num, start=start+1, searchType='image')

def getImageResults(term, num, max_num=10):
    ret = []
    while len(ret) < num:
        cur_num = min(num - len(ret), max_num)
        results = getImageResultsInner(term, cur_num, len(ret))
        imgs = imgsFromResults(results)
        ret.extend(imgs)
        print('n=', len(ret))
    assert len(ret) == num
    return ret
    

def makeSquare(img):
    h, w = img.shape[:2]

    yc = h//2
    xc = w//2

    sz = min(h, w)
    step = sz//2

    return img[yc-step:yc+step,xc-step:xc+step,:]

def makeVideoFromImgs(fname, imgs, fps): 
    
    video_dim = (1000, 1000)
    video_fps = fps
    vidwriter = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*"avc1"), video_fps, video_dim)
    
    for img in imgs:
        scaled = cv2.resize(img, dsize=video_dim, interpolation=cv2.INTER_LINEAR)
        vidwriter.write(scaled)
    vidwriter.release()
