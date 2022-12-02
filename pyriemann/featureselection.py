import numpy as np
from .utils.distance import distance_riemann
from .utils.mean import mean_riemann


def class_distinctiveness(C_class1, C_class2):
    """Measure class separability between two classes on a manifold.

        Parameters
        ----------
        C_class1 : ndarray, shape (..., n, n)
            Input matrices from class1.
        C_class1 : ndarray, shape (..., n, n)
            Input matrices from class2.

        Returns
        -------
        class_dis : float
            ClassDis value between class1 and class2
        numerator : float
            Numerator value of ClassDis
        denominator : float
            Denominator value of classDis

        References
        ----------
        .. [1] `Defining and quantifying users’ mental imagery-based BCI skills: a first step
        <https://iopscience.iop.org/article/10.1088/1741-2552/aac577/meta?casa_token=l7DBy6pHBwAAAAAA:AOA8jG2Qgqh2YYgXojHUhDdLEpy4-lWBrHnjL-wurpkvwkzQOzgbzP2b0LqOiop4R6nWU_I_0t8v>`_
        F. Lotte, and C. Jeunet. Journal of neural engineering, 15(4), 046030, 2018.
        """

    #numerator computation
    mean_class1 = mean_riemann(C_class1)
    mean_class2 = mean_riemann(C_class2)
    numerator = distance_riemann(mean_class1, mean_class2)

    #denominator computation
    dis_within_class1 = [distance_riemann(C_class1[r, :, :], mean_class1) for r in range(len(C_class1))]
    dis_within_class2 = [distance_riemann(C_class2[r, :, :], mean_class2) for r in range(len(C_class2))]
    sigma_class1 = np.sum(dis_within_class1) / len(C_class1)
    sigma_class2 = np.sum(dis_within_class2) / len(C_class2)
    denominator = 0.5 * (sigma_class1 + sigma_class2)

    class_dis = numerator / denominator

    return class_dis, numerator, denominator
