import math
import cv2
import numpy as np
import imageio


def makeDctBasis() :
    N = 8
    M = 8
    basis = [list() for _ in range(N)]
    for u in range(N) :
        for v in range(M) :
            tmp = [[0]*N for _ in range(N)]
            Cu = 1 if u != 0 else (2 ** 0.5) / 2
            Cv = 1 if v != 0 else (2 ** 0.5) / 2
            constant = (Cu * Cv) / 4
            for i in range(N) :
                for j in range(M) :
                    cosU = math.cos(((2*i+1) * (u*math.pi))/(N*2))
                    cosV = math.cos(((2*j+1) * (v*math.pi))/(N*2))
                    tmp[i][j] = (constant*cosU*cosV)
            basis[u].append(tmp)
    return basis

def makeIDctBasis():
    N = 8
    M = 8
    basis = [list() for _ in range(N)]
    for i in range(N):
        for j in range(M):
            tmp = [[0] * N for _ in range(N)]
            for u in range(N):
                for v in range(M):
                    Cu = 1 if u != 0 else (2 ** 0.5) / 2
                    Cv = 1 if v != 0 else (2 ** 0.5) / 2
                    constant = (Cu * Cv) / 4
                    cosU = math.cos(((2 * i + 1) * (u * math.pi)) / (N * 2))
                    cosV = math.cos(((2 * j + 1) * (v * math.pi)) / (N * 2))
                    tmp[u][v] = (constant * cosU * cosV)
            basis[i].append(tmp)
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

def inverseDctPerform(basis,blocks) :
    N = 8
    for u in range(len(blocks)) :
        for v in range(len(blocks[0])) :
            fBlock = list([0] * N for _ in range(N))
            for bi in range(len(basis)) :
                for bj in range(len(basis[0])) :
                    b = basis[bi][bj]
                    accum = 0
                    for i in range(N) :
                        for j in range(N) :
                            accum += (blocks[u][v][i][j] * b[i][j])
                    fBlock[bi][bj] = round((accum*1000))
            blocks[u][v] = fBlock
    return blocks

def convertListToNumpyArary(blocks) :
    blocks = np.array(blocks)
    for i in range(len(blocks)) :
        blocks[i] = np.array(blocks[i])
        for j in range(len(blocks[0])) :
            blocks[i][j] = np.array(blocks[i][j])
    return blocks

def mergeBlocks(blocks,h,w) :
    N = 8
    arr = np.full((h, w),0)
    for u in range(len(blocks)) :
        for v in range(len(blocks[0])) :
            for i in range(N) :
                for j in range(N) :
                    arr[i+(u*N)][j+(v*N)] = int(blocks[u][v][i][j])
    return arr
    # for i in range(len(blocks)) :
    #     merged = np.vstack((merged,blocks[i]))
    # print(merged)
        # for j in range(len(blocks[0])) :
        #     print(blocks[i][j])
        #     print()
        #     print()


if __name__ == "__main__" :

    # 1 - 2D DCT에 사용될 8x8 DCT basis 구하기
    basis = makeDctBasis()

    # 2 - 너무 크지 않은 영상 (ex. Lena)을 read 하여 graylevel로 변환
    img = cv2.imread("lena.png")
    h, w, _ = img.shape
    grayImg = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    # 3 - 이미지 blocks으로 나누기 & DCT 계산 후, Fuv Blocks 반환
    blocks = getBlocks(grayImg,h,w)
    FuvBlocks = dctPerform(basis,blocks)

    # 4 - F(u,v)를 다시 IDCT로 8x8 영상을 다시 복원
    # 4-1) F(u,v) 그대로 사용
    basis = makeIDctBasis()
    fijBlocks = inverseDctPerform(basis,FuvBlocks)
    fijBlocks = convertListToNumpyArary(fijBlocks)
    fij = mergeBlocks(fijBlocks,h,w)
    cv2.imwrite('test1.jpeg', fij)
    # 4-2) F(u,v) 좌상단 4x4는 그대로 두고 나머지는 0로 변경 사용






