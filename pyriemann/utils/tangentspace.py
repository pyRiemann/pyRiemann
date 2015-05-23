import numpy
from .base import sqrtm, invsqrtm, logm, expm

###############################################################
# Tangent Space
###############################################################


def tangent_space(covmats, Cref):
    """Project a set of covariance matrices in the tangent space according to the given reference point Cref

    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param Cref: The reference covariance matrix
    :returns: the Tangent space , a matrix of Ntrials X (Nchannels*(Nchannels+1)/2)

    """
    Nt, Ne, Ne = covmats.shape
    Cm12 = invsqrtm(Cref)
    idx = numpy.triu_indices_from(Cref)
    T = numpy.empty((Nt, Ne * (Ne + 1) / 2))
    coeffs = (
        numpy.sqrt(2) *
        numpy.triu(
            numpy.ones(
                (Ne,
                 Ne)),
            1) +
        numpy.eye(Ne))[idx]
    for index in range(Nt):
        tmp = numpy.dot(numpy.dot(Cm12, covmats[index, :, :]), Cm12)
        tmp = logm(tmp)
        T[index, :] = numpy.multiply(coeffs, tmp[idx])
    return T


def untangent_space(T, Cref):
    """Project a set of Tangent space vectors in the manifold according to the given reference point Cref

    :param T: the Tangent space , a matrix of Ntrials X (Nchannels*(Nchannels+1)/2)
    :param Cref: The reference covariance matrix
    :returns: A set of Covariance matrix, Ntrials X Nchannels X Nchannels

    """
    Nt, Nd = T.shape
    Ne = int((numpy.sqrt(1 + 8 * Nd) - 1) / 2)
    C12 = sqrtm(Cref)

    idx = numpy.triu_indices_from(Cref)
    covmats = numpy.empty((Nt, Ne, Ne))
    covmats[:, idx[0], idx[1]] = T
    for i in range(Nt):
        covmats[i] = numpy.diag(numpy.diag(covmats[i])) + numpy.triu(
            covmats[i], 1) / numpy.sqrt(2) + numpy.triu(covmats[i], 1).T / numpy.sqrt(2)
        covmats[i] = expm(covmats[i])
        covmats[i] = numpy.dot(numpy.dot(C12, covmats[i]), C12)

    return covmats
