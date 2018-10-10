import collections
import itertools
import linecache
import sys
import traceback


# the classes and functions in this module are almost equivalent to (most code is copied as-is from) the
# corresponding class/function from the traceback module in the stdlib. The only significant difference (except from
# simplification thanks to only supporting the options we need) is locals are stored as-is in the FrameSummary
# instead of as a dict of repr.
class FrameSummary(object):
    """A single frame from a traceback.

    Attributes
    ----------
    filename : str
        The filename for the frame.
    lineno : int
        The line within filename for the frame that was active when the frame was captured.
    name : str
        The name of the function or method that was executing when the frame was captured.
    line : str
        The text from the linecache module for the code that was running when the frame was captured.
    locals : dict
        The frame locals, which are stored as-is.

    Notes
    -----
    equivalent to traceback.FrameSummary except locals are stored as-is instead of their repr.
    """

    __slots__ = ('filename', 'lineno', 'name', 'locals', '_line')

    def __init__(self, filename, lineno, name, locals):
        """Construct a FrameSummary.

        Parameters
        ----------
        filename : str
            The filename for the frame.
        lineno : int
            The line within filename for the frame that was active when the frame was captured.
        name : str
            The name of the function or method that was executing when the frame was captured.
        locals : dict
            The frame locals, which are stored as-is.
        """
        self.filename = filename
        self.lineno = lineno
        self.name = name
        self.locals = locals
        self._line = None

    @property
    def line(self):
        if self._line is None:
            self._line = linecache.getline(self.filename, self.lineno).strip()
        return self._line


class StackSummary(list):
    @classmethod
    def extract(klass, frame_gen, limit=None):
        """Create a StackSummary from an iterable of frames.

        Parameters
        ----------
        frame_gen : generator
            A generator that yields (frame, lineno) tuples to include in the stack summary.
        limit : int, optional
            Number of frames to include. Defaults to None (include all frames).

        Notes
        -----
        This is almost equivalent to (the code is mostly copied from)

            traceback.StackSummary.extract(frame_gen, limit=limit, lookup_lines=False, capture_locals=True)

        but the extracted locals are the actual dict instead of a repr of it.
        """
        if limit is None:
            limit = getattr(sys, 'tracebacklimit', None)
            if limit is not None and limit < 0:
                limit = 0
        if limit is not None:
            if limit >= 0:
                frame_gen = itertools.islice(frame_gen, limit)
            else:
                frame_gen = collections.deque(frame_gen, maxlen=-limit)

        result = klass()
        filenames = set()
        for frame, lineno in frame_gen:
            f_code = frame.f_code
            filename = f_code.co_filename
            filenames.add(filename)

            # actual line lookups will happen lazily.
            # f_globals is necessary for atypical modules where source must be fetched via module.__loader__.get_source
            linecache.lazycache(filename, frame.f_globals)

            summary = FrameSummary(filename=filename, lineno=lineno, name=f_code.co_name,
                                   locals=frame.f_locals)
            result.append(summary)

        # Discard cache entries that are out of date.
        for filename in filenames:
            linecache.checkcache(filename)
        return result


def extract_stack(frame, limit=None):
    """Extract the raw traceback from the current stack frame.

    The return value has the same format as for extract_tb().  The
    optional 'f' and 'limit' arguments have the same meaning as for
    print_stack().  Each item in the list is a quadruple (filename,
    line number, function name, text), and the entries are in order
    from oldest to newest stack frame.
    """
    stack = StackSummary.extract(traceback.walk_stack(frame), limit=limit)
    stack.reverse()
    return stack


def extract_tb(tb, limit=None):
    """
    Return a StackSummary object representing a list of
    pre-processed entries from traceback.

    This is useful for alternate formatting of stack traces.  If
    'limit' is omitted or None, all entries are extracted.  A
    pre-processed stack trace entry is a FrameSummary object
    containing attributes filename, lineno, name, and line
    representing the information that is usually printed for a stack
    trace.  The line is a string with leading and trailing
    whitespace stripped; if the source is not available it is None.
    """
    return StackSummary.extract(traceback.walk_tb(tb), limit=limit)
