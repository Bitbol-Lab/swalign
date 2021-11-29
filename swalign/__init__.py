#!/usr/bin/env python
"""
Simple Smith-Waterman aligner
"""
import sys
from io import StringIO

import numpy as np
import numba as nb


class ScoringMatrix(object):
    """
    Read scoring matrix from a file or string

    Matrix should be space-delimited in a format like:

      A C G T
    A 1 0 0 0
    C 0 1 0 0
    G 0 0 1 0
    T 0 0 0 1

    Rows and Columns must be in the same order

    """

    def __init__(self, filename=None, text=None, wildcard_score=0):
        assert filename or text

        if filename:
            fs = open(filename)
        else:
            fs = StringIO.StringIO(text)

        self.scores = []
        self.bases = None
        self.wildcard_score = wildcard_score

        for line in fs:
            if line[0] == '#' or not line.strip():
                continue

            if not self.bases:
                self.bases = line.split()
                self.base_count = len(self.bases)
            else:
                cols = line.split()
                self.scores.extend([float(x) for x in cols[1:]])

        fs.close()

    def score(self, one, two, wildcard=None):
        if self.wildcard_score and wildcard and (one in wildcard or two in wildcard):
            return self.wildcard_score

        one_idx = 0
        two_idx = 0
        for i, b in enumerate(self.bases):
            if b == one:
                one_idx = i
            if b == two:
                two_idx = i

        return self.scores[(one_idx * self.base_count) + two_idx]


class IdentityScoringMatrix(object):
    def __init__(self, match=1, mismatch=-1):
        self.match = match
        self.mismatch = mismatch

    def score(self, one, two, wildcard=None):
        if wildcard and (one in wildcard or two in wildcard):
            return self.match

        if one == two:
            return self.match
        return self.mismatch


NucleotideScoringMatrix = IdentityScoringMatrix


class Matrix(object):
    def __init__(self, rows, cols, init=None):
        self.rows = rows
        self.cols = cols
        self.values = [init, ] * rows * cols

    def get(self, row, col):
        return self.values[(row * self.cols) + col]

    def set(self, row, col, val):
        self.values[(row * self.cols) + col] = val


@nb.njit
def _score(match, mismatch, one, two, wildcard=None):
    if (wildcard is not None) and (one in wildcard or two in wildcard):
        return match

    if one == two:
        return match
    return mismatch


@nb.njit
def _align(ref, query, match, mismatch, gap_penalty, gap_extension_penalty, gap_extension_decay, wildcard, globalalign,
           full_query, prefer_gap_runs):
    n_rows, n_cols = len(query) + 1, len(ref) + 1

    matrix_0 = np.zeros((n_rows, n_cols), dtype=np.int64)
    matrix_1 = np.full((n_rows, n_cols), fill_value=" ")
    matrix_2 = np.zeros((n_rows, n_cols), dtype=np.int64)
    for row in range(1, n_rows):
        matrix_0[row, 0] = 0
        matrix_1[row, 0] = "i"
        matrix_2[row, 0] = 0

    for col in range(1, n_cols):
        matrix_0[0, col] = 0
        matrix_1[0, col] = "d"
        matrix_2[0, col] = 0

    max_val = 0
    max_row = 0
    max_col = 0

    # calculate matrix
    for row in range(1, n_rows):
        for col in range(1, n_cols):
            mm_val = matrix_0[row - 1, col - 1] + _score(match, mismatch, query[row - 1], ref[col - 1], wildcard)

            ins_run = 0
            del_run = 0

            if matrix_1[row - 1, col] == "i":
                ins_run = matrix_2[row - 1, col]
                if not matrix_0[row - 1, col]:
                    # no penalty to start the alignment
                    ins_val = 0
                else:
                    if not gap_extension_decay:
                        ins_val = matrix_0[row - 1, col] + gap_extension_penalty
                    else:
                        ins_val = matrix_0[row - 1, col] + min(0, gap_extension_penalty + ins_run * gap_extension_decay)
            else:
                ins_val = matrix_0[row - 1, col] + gap_penalty

            if matrix_1[row, col - 1] == 'd':
                del_run = matrix_2[row, col - 1]
                if not matrix_0[row, col - 1]:
                    # no penalty to start the alignment
                    del_val = 0
                else:
                    if not gap_extension_decay:
                        del_val = matrix_0[row, col] + gap_extension_penalty
                    else:
                        del_val = matrix_0[row, col - 1] + min(0, gap_extension_penalty + del_run * gap_extension_decay)

            else:
                del_val = matrix_0[row, col - 1] + gap_penalty

            if globalalign or full_query:
                cell_val = max(mm_val, del_val, ins_val)
            else:
                cell_val = max(mm_val, del_val, ins_val, 0)

            if not prefer_gap_runs:
                ins_run = 0
                del_run = 0

            if del_run and cell_val == del_val:
                val_0, val_1, val_2 = cell_val, "d", del_run + 1
            elif ins_run and cell_val == ins_val:
                val_0, val_1, val_2 = cell_val, "i", ins_run + 1
            elif cell_val == mm_val:
                val_0, val_1, val_2 = cell_val, "m", 0
            elif cell_val == del_val:
                val_0, val_1, val_2 = cell_val, "d", 1
            elif cell_val == ins_val:
                val_0, val_1, val_2 = cell_val, "i", 1
            else:
                val_0, val_1, val_2 = 0, "x", 0

            if val_0 >= max_val:
                max_val = val_0
                max_row = row
                max_col = col

            matrix_0[row, col] = val_0
            matrix_1[row, col] = val_1
            matrix_2[row, col] = val_2

    # backtrack
    if globalalign:
        # backtrack from last cell
        row = n_rows - 1
        col = n_cols - 1
    elif full_query:
        # backtrack from max in last row
        row = n_rows - 1
        max_val = 0
        for c in range(1, n_cols):
            if matrix_0[row, c] > max_val:
                max_val = matrix_0[row, c]
        col = n_cols - 1
    else:
        # backtrack from max
        row = max_row
        col = max_col

    aln = []
    path = []
    while True:
        val, op = matrix_0[row, col], matrix_1[row, col]

        if globalalign:
            if (not row) and (not col):
                break
        elif full_query:
            if not row:
                break
        elif val <= 0:
            break

        path.append((row, col))
        aln.append(op)

        if op == "m":
            row -= 1
            col -= 1
        elif op == "i":
            row -= 1
        elif op == "d":
            col -= 1
        else:
            break

    aln = aln[::-1]

    return aln, row, col, max_row, max_col, max_val, path, matrix_0, matrix_1, matrix_2


class LocalAlignment(object):
    def __init__(self, scoring_matrix, gap_penalty=-1, gap_extension_penalty=-1, gap_extension_decay=0.0,
                 prefer_gap_runs=True, verbose=False, globalalign=False, wildcard=None, full_query=False):
        self.scoring_matrix = scoring_matrix
        self.gap_penalty = gap_penalty
        self.gap_extension_penalty = gap_extension_penalty
        self.gap_extension_decay = gap_extension_decay
        self.verbose = verbose
        self.prefer_gap_runs = prefer_gap_runs
        self.globalalign = globalalign
        self.wildcard = wildcard
        self.full_query = full_query

    def align(self, ref, query, ref_name='', query_name='', rc=False):
        orig_ref = ref
        orig_query = query

        ref = ref.upper()
        query = query.upper()

        aln, row, col, max_row, max_col, max_val, path, matrix_0, matrix_1, matrix_2 = _align(
            ref, query, self.scoring_matrix.match, self.scoring_matrix.mismatch, self.gap_penalty,
            self.gap_extension_penalty, self.gap_extension_decay, self.prefer_gap_runs, self.globalalign, self.wildcard,
            self.full_query
            )

        if self.verbose:
            n_rows, n_cols = len(query) + 1, len(ref) + 1
            matrix = Matrix(n_rows, n_cols)
            for row in range(n_rows):
                for col in range(n_cols):
                    matrix.set(row, col, (matrix_0[row, col], matrix_1[row, col], matrix_2[row, col]))
            self.dump_matrix(ref, query, matrix, path)
            print(aln)
            print((max_row, max_col), max_val)

        cigar = _reduce_cigar(aln)
        return Alignment(orig_query, orig_ref, row, col, cigar, max_val, ref_name, query_name, rc, self.globalalign,
                         self.wildcard)

    @staticmethod
    def dump_matrix(ref, query, matrix, path, show_row=-1, show_col=-1):
        sys.stdout.write('      -      ')
        sys.stdout.write('       '.join(ref))
        sys.stdout.write('\n')
        for row in range(matrix.rows):
            if row == 0:
                sys.stdout.write('-')
            else:
                sys.stdout.write(query[row - 1])

            for col in range(matrix.cols):
                if show_row == row and show_col == col:
                    sys.stdout.write('       *')
                else:
                    sys.stdout.write(' %5s%s%s' % (matrix.get(row, col)[0],
                                                   matrix.get(row, col)[1], '$' if (row, col) in path else ' '))
            sys.stdout.write('\n')


def _reduce_cigar(operations):
    count = 1
    last = None
    ret = []
    for op in operations:
        if last and op == last:
            count += 1
        elif last:
            ret.append((count, last.upper()))
            count = 1
        last = op

    if last:
        ret.append((count, last.upper()))
    return ret


def _cigar_str(cigar):
    out = ''
    for num, op in cigar:
        out += '%s%s' % (num, op)
    return out


class Alignment(object):
    def __init__(self, query, ref, q_pos, r_pos, cigar, score, ref_name='', query_name='', rc=False, globalalign=False,
                 wildcard=None):
        self.query = query
        self.ref = ref
        self.q_pos = q_pos
        self.r_pos = r_pos
        self.cigar = cigar
        self.score = score
        self.r_name = ref_name
        self.q_name = query_name
        self.rc = rc
        self.globalalign = globalalign
        self.wildcard = wildcard

        self.r_offset = 0
        self.r_region = None

        self.orig_query = query
        self.query = query.upper()

        self.orig_ref = ref
        self.ref = ref.upper()

        q_len = 0
        r_len = 0

        self.matches = 0
        self.mismatches = 0

        i = self.r_pos
        j = self.q_pos

        for count, op in self.cigar:
            if op == 'M':
                q_len += count
                r_len += count
                for k in range(count):
                    if self.query[j] == self.ref[i]:
                        self.matches += 1
                    else:
                        self.mismatches += 1
                    i += 1
                    j += 1

            elif op == 'I':
                q_len += count
                j += count
                self.mismatches += count
            elif op == 'D':
                r_len += count
                i += count
                self.mismatches += count

        self.q_end = q_pos + q_len
        self.r_end = r_pos + r_len
        if self.mismatches + self.matches > 0:
            self.identity = float(self.matches) / (self.mismatches + self.matches)
        else:
            self.identity = 0

    def set_ref_offset(self, ref, offset, region):
        self.r_name = ref
        self.r_offset = offset
        self.r_region = region

    @property
    def extended_cigar_str(self):
        qpos = 0
        rpos = 0
        ext_cigar_str = ''
        working = []
        for count, op in self.cigar:
            if op == 'M':
                for k in range(count):
                    if self.query[self.q_pos + qpos + k] == self.ref[self.r_pos + rpos + k]:
                        ext_cigar_str += 'M'
                    else:
                        ext_cigar_str += 'X'
                qpos += count
                rpos += count

            elif op == 'I':
                qpos += count
                ext_cigar_str += 'I' * count
            elif op == 'D':
                rpos += count
                ext_cigar_str += 'D' * count

            working = _reduce_cigar(ext_cigar_str)

        out = ''
        for num, op in working:
            out += '%s%s' % (num, op)
        return out

    @property
    def cigar_str(self):
        return _cigar_str(self.cigar)

    def dump(self, wrap=None, out=sys.stdout):
        i = self.r_pos
        j = self.q_pos

        q = ''
        m = ''
        r = ''
        qlen = 0
        rlen = 0

        for count, op in self.cigar:
            if op == 'M':
                qlen += count
                rlen += count
                for k in range(count):
                    q += self.orig_query[j]
                    r += self.orig_ref[i]
                    if self.query[j] == self.ref[i] or (self.wildcard and (self.query[j] in self.wildcard or self.ref[i] in self.wildcard)):
                        m += '|'
                    else:
                        m += '.'

                    i += 1
                    j += 1
            elif op == 'D':
                rlen += count
                for k in range(count):
                    q += '-'
                    r += self.orig_ref[i]
                    m += ' '
                    i += 1
            elif op == 'I':
                qlen += count
                for k in range(count):
                    q += self.orig_query[j]
                    r += '-'
                    m += ' '
                    j += 1

            elif op == 'N':
                q += '-//-'
                r += '-//-'
                m += '    '

        if self.q_name:
            out.write('Query: %s%s (%s nt)\n' % (self.q_name, ' (reverse-compliment)' if self.rc else '', len(self.query)))
        if self.r_name:
            if self.r_region:
                out.write('Ref  : %s (%s)\n\n' % (self.r_name, self.r_region))
            else:
                out.write('Ref  : %s (%s nt)\n\n' % (self.r_name, len(self.ref)))

        poslens = [self.q_pos + 1, self.q_end + 1, self.r_pos + self.r_offset + 1, self.r_end + self.r_offset + 1]
        maxlen = max([len(str(x)) for x in poslens])

        q_pre = 'Query: %%%ss ' % maxlen
        r_pre = 'Ref  : %%%ss ' % maxlen
        m_pre = ' ' * (8 + maxlen)

        rpos = self.r_pos
        if not self.rc:
            qpos = self.q_pos
        else:
            qpos = self.q_end

        while q and r and m:
            if not self.rc:
                out.write(q_pre % (qpos + 1))  # pos is displayed as 1-based
            else:
                out.write(q_pre % (qpos))  # revcomp is 1-based on the 3' end

            if wrap:
                qfragment = q[:wrap]
                mfragment = m[:wrap]
                rfragment = r[:wrap]

                q = q[wrap:]
                m = m[wrap:]
                r = r[wrap:]
            else:
                qfragment = q
                mfragment = m
                rfragment = r

                q = ''
                m = ''
                r = ''

            out.write(qfragment)
            if not self.rc:
                for base in qfragment:
                    if base != '-':
                        qpos += 1
            else:
                for base in qfragment:
                    if base != '-':
                        qpos -= 1

            if not self.rc:
                out.write(' %s\n' % qpos)
            else:
                out.write(' %s\n' % (qpos + 1))

            out.write(m_pre)
            out.write(mfragment)
            out.write('\n')
            out.write(r_pre % (rpos + self.r_offset + 1))
            out.write(rfragment)
            for base in rfragment:
                if base != '-':
                    rpos += 1
            out.write(' %s\n\n' % (rpos + self.r_offset))

        out.write("Score: %s\n" % self.score)
        out.write("Matches: %s (%.1f%%)\n" % (self.matches, self.identity * 100))
        out.write("Mismatches: %s\n" % (self.mismatches,))
        out.write("CIGAR: %s\n" % self.cigar_str)


def fasta_gen(fname):
    def gen():
        seq = ''
        name = ''
        comments = ''

        if fname == '-':
            f = sys.stdin
            name = 'stdin'
        else:
            f = open(fname)

        for line in f:
            if line[0] == '>':
                if name and seq:
                    yield (name, seq, comments)

                spl = line[1:].strip().split(' ', 1)
                name = spl[0]
                if len(spl) > 1:
                    comments = spl[1]
                else:
                    comments = ''

                seq = ''
            else:
                seq += line.strip()

        if name and seq:
            yield (name, seq, comments)

        if fname != '-':
            f.close()
    return gen


def seq_gen(name, seq):
    def gen():
        yield (name, seq, '')

    return gen


def extract_region(comments):
    ref = None
    start = None
    # start_offset = 0
    # end_offset = 0

    try:
        attrs = comments.split(' ')
        for attr in attrs:
            if '=' in attr:
                k, v = attr.split('=')
                if k == 'range':
                    spl = v.split(':')
                    ref = spl[0]
                    start, end = [int(x) for x in spl[1].split('-')]
                # elif k == "5'pad":
                #     start_offset = int(v)
                # elif k == "3'pad":
                #     end_offset = int(v)
    except:
        pass

    if ref and start:
        return (ref, start - 1, '%s:%s-%s' % (ref, start, end))

    return None


__revcomp = {}
for a, b in zip('atcgATCGNn', 'tagcTAGCNn'):
    __revcomp[a] = b
__cache = {}


def revcomp(seq):
    if seq in __cache:
        return __cache[seq]

    ret = []
    for s in seq.upper()[::-1]:
        ret.append(__revcomp[s])

    __cache[seq] = ''.join(ret)
    return __cache[seq]


#     sw.align('ACACACTA','AGCACACA').dump()
#     aln=sw.align("AAGGGGAGGACGATGCGGATGTTC","AGGGAGGACGATGCGG")
#     aln.dump()
