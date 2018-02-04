import numpy as np

if __name__ == "__main__":
    a = np.zeros((3,3), dtype=np.uint32)
    x = np.array([-1,2,3])
    y = np.array([2,1,0])

    yx = np.column_stack((y, x))
    #yx2 = yx[yx[:,1]<=2]
    #yx2 = yx[yx[:,1]<=2 and yx[:,1]>=0]
    #yx2 = yx[0<=yx[:,1]<=2]
    #yx2 = yx[0<=yx[:,1] & yx[:,1]<=2]
    yx2 = yx[np.logical_and(0<=yx[:,1], yx[:,1]<=2)]
    #yx2 = yx[yx[:,1] in range(0,3)]

    print("a")
    print(a)

    print("yx")
    print(yx)


    print("yx2")
    print(yx2)

    y2 = yx2[:, 0]
    x2 = yx2[:, 1]

    a[y2,x2] = 1
    print("a")
    print(a)

    print(np.ravel(list(zip([1,3,5], [2,4,6]))))