


class cls1(object):

    def run(self, i):
        print "I am class 1 with", i


class cls2(cls1):

    def run(self, j, k=3):
        print "I am class 2 with", j, "and", k


if __name__ == "__main__":
    c = cls2()
    c.run(3)