# Problem 1
# hello.py

#!/usr/bin/env python
import numpy as np
import cv2
from matplotlib import pyplot as plt

# import modules used here -- sys is a very standard one
import sys
# Gather our code in a main() function

def main():
    print('Hello there {}'.format(sys.argv[1]))
    # Command line args are in sys.argv[1], sys.argv[2] ...
    # sys.argv[0] is the script name itself and can be ignored
    img = cv2.imread('samoyed.jpg',1)
    # Display with matplotlib
    plt.imshow(img, interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    # Close the window will exit the program
    cv2.destroyAllWindows()
    # Standard boilerplate to call the main() function to begin
    # the program.

#----------------------------------------------------------------------------
# Problem 2
def twoIndep():
    # matrix A
    A = [[4.29,2.2,5.51],
        [5.20,10.1,-8.24],
        [1.33,4.8,-6.62]]

    # determinant of A
    detA = np.linalg.det(A)
    str1 = 'The determinant of matrix A is ' + repr(detA)
    print(str1)

    # singular value decomposition
    U, s, V = np.linalg.svd(A)
    str2 = 's = ' + repr(s)
    print(str2)

    # ranks of matrix A and matrix U
    rankA = np.linalg.matrix_rank(A)
    m, n = U.shape
    str3 = 'rank of A = ' + repr(rankA)
    str4 = 'number of columns in U = ' + repr(n)
    print(str3)
    print(str4)

    if rankA == n:
        print("All columns are linearly independent.")
    else:
        print("There is at least one linearly dependent column")

#----------------------------------------------------------------------------
# Problem 3
#!/usr/bin/env python
import numpy as np
import cv2
from matplotlib import pyplot as plt

# import modules used here -- sys is a very standard one
import sys

# Gather our code in a lse() function
def lse():
    x = np.matrix([0, 1, 1.9, 3, 3.9, 5])
    y = np.matrix([1, 3.2, 5, 7.2, 9.3, 11.1])
    A = np.matrix([[1,0],[1,1],[1,1.9],[1,3],[1,3.9],[1,5]])
    b = y.T
    q = np.linalg.inv((A.T).dot(A)).dot(A.T).dot(b)

    s = 'This is q : ' + repr(q)
    print(s)

if __name__ == '__main__':
    main()
    twoIndep()
    lse()