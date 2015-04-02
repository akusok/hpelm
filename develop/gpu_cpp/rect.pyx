

cdef extern from "Rectangle.h" namespace "shapes":
    cdef cppclass Rectangle:
        Rectangle(int, int, int, int) except +
        int x0, y0, x1, y1
        int getLength()
        int getHeight()
        int getArea()
        void move(int, int)


def lol():
    cdef Rectangle *rec = new Rectangle(1, 2, 3, 4)
    try:
        recLength = rec.getLength()
        print recLength
    finally:
        del rec     # delete heap allocated object
