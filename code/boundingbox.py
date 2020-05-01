

class Boundingbox():

#   (x1)############(x2)
    #                 #
    #                 #
    #                 #
    #                 #
    #                 #
#   (y1)#############(y2)


    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.width = abs(xmin - xmax)
        self.height = abs(ymin - ymax)
        self.area = self.width * self.height


