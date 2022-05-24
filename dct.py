import math
import cv2
import numpy as np


def makeDctBasis() :
    N = 8
    M = 8
    basis = [list() for _ in range(N)]
    for u in range(N) :
        for v in range(M) :
            tmp = [[0]*N for _ in range(N)]
            for i in range(N) :
                for j in range(M) :
                    Cu = 1 if u != 0 else (2**0.5)/2
                    Cv = 1 if v != 0 else (2**0.5)/2
                    constant = (Cu*Cv)/4
                    cosU = math.cos(((2*i+1) * (u*math.pi))/(N*2))
                    cosV = math.cos(((2*j+1) * (v*math.pi))/(N*2))
                    tmp[i][j] = (constant*cosU*cosV)
            basis[u].append(tmp)
    return basis

    # for i in range(len(grayImg)) :
    #     for j in range(len(grayImg[0])) :
    #         print(grayImg[i][j] , end = " ")
    #     print()

def getBlocks(img,h,w) :
    blocks = [[0] * (w // 8) for _ in range(h // 8)]
    splitedImg = np.vsplit(img,h//8)
    for idx1,ary in enumerate(splitedImg) :
        nary = np.hsplit(ary, w // 8)
        for idx2,block in enumerate(nary) :
            blocks[idx1][idx2] = block

    return blocks

    # for i in range(N) :
    #     for j in range(M) :
    #         res = 0
    #         # print(len(block),len(block[0]))
    #         # print(len(basis[i][j]),len(basis[i][j][0]))
    #         for q in range(N) :
    #             for w in range(M):
    #                 res += ( block[q][w] * basis[i][j][q][w] )
    #         print((i,j),end =" : ")
    #         print(res)


def dctPerform(basis,blocks) :
    N = 8
    for u in range(len(blocks)) :
        for v in range(len(blocks[0])) :
            Fblock = list([0] * N for _ in range(N))
            for bi in range(len(basis)) :
                for bj in range(len(basis[0])) :
                    b = basis[bi][bj]
                    accum = 0
                    for i in range(N) :
                        for j in range(N) :
                            accum += (blocks[u][v][i][j] * b[i][j])
                    Fblock[bi][bj] = round((accum/1000),4)
            blocks[u][v] = Fblock
    return blocks



if __name__ == "__main__" :
    # 1
    basis = makeDctBasis()
    # 2
    img = cv2.imread("lena.png")
    h, w, _ = img.shape
    grayImg = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # cv2.imshow('dasd ', grayImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 3 - 이미지 blocks으로 나누기 & Fuv 계산
    blocks = getBlocks(grayImg,h,w)
    Fuv = dctPerform(basis,blocks)


# blocks = [[[[182,196,199,201,203,201,199,173],[175,180,176,142,148,152,148,120],[148,118,123,115,114,107,108,107],[115,110,110,112,105,109,101,100],[104,106,106,102,104,95,98,105],[99,115,131,104,118,86,87,133],[112,154,154,107,140,97,88,151],[145,158,178,123,132,140,138,133]]]]