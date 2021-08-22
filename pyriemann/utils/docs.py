import warnings


class deprecated(object):
    """Mark a function or class as deprecated (decorator).

    Issue a warning when the function is called/the class is instantiated and
    adds a warning to the docstring.

    The optional extra argument will be appended to the deprecation message
    and the docstring. Note: to use this with the default value for extra, put
    in an empty of parentheses::

        >>> from pyriemann.utils import deprecated
        >>> deprecated()
        <pyriemann.utils.docs.deprecated object at ...>
        >>> @deprecated()
        ... def some_function(): pass


    Parameters
    ----------
    extra: string
        To be added to the deprecation messages.
    """

    # Borrowed from MNE:
    # https://mne.tools/stable/generated/mne.utils.deprecated.html

    def __init__(self, extra=""):
        self.extra = extra

    def __call__(self, obj):
        """Call.
        Parameters
        ----------
        obj : object
            Object to call.
        """
        if isinstance(obj, type):
            return self._decorate_class(obj)
        else:
            return self._decorate_fun(obj)

    def _decorate_class(self, cls):
        msg = "Class %s is deprecated" % cls.__name__
        if self.extra:
            msg += "; %s" % self.extra

        # FIXME: we should probably reset __new__ for full generality
        init = cls.__init__

        def deprecation_wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return init(*args, **kwargs)

        cls.__init__ = deprecation_wrapped

        deprecation_wrapped.__name__ = "__init__"
        deprecation_wrapped.__doc__ = self._update_doc(init.__doc__)
        deprecation_wrapped.deprecated_original = init

        return cls

    def _decorate_fun(self, fun):
        """Decorate function fun."""
        msg = "Function %s is deprecated" % fun.__name__
        if self.extra:
            msg += "; %s" % self.extra

        def deprecation_wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return fun(*args, **kwargs)

        deprecation_wrapped.__name__ = fun.__name__
        deprecation_wrapped.__dict__ = fun.__dict__
        deprecation_wrapped.__doc__ = self._update_doc(fun.__doc__)

        return deprecation_wrapped

    def _update_doc(self, olddoc):
        newdoc = ".. warning:: DEPRECATED"
        if self.extra:
            newdoc = "%s: %s" % (newdoc, self.extra)
        if olddoc:
            # Get the spacing right to avoid sphinx warnings
            n_space = 4
            for li, line in enumerate(olddoc.split("\n")):
                if li > 0 and len(line.strip()):
                    n_space = len(line) - len(line.lstrip())
                    break
            newdoc = "%s\n\n%s%s" % (newdoc, " " * n_space, olddoc)

        return newdoc
