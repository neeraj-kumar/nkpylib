"""
Various NLP-related utilities.

Licensed under the 3-clause BSD License:

Copyright (c) 2013, Neeraj Kumar (neerajkumar.org)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the author nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL NEERAJ KUMAR BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os, sys, time
from nkutils import *

STOP_WORDS = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])

def fullnlp(s):
    """Tokenizes, tags, and runs named entity recognition on an input string.
    Returns tokens, tagged, entities, flatentities, candidates"""
    import nltk
    tokens = nltk.word_tokenize(s)
    tagged = nltk.pos_tag(tokens)
    entities = nltk.chunk.ne_chunk(tagged)
    flatents = [x for x in entities if not isinstance(x, tuple)]
    flatents = [(c.node, [l for l in c.leaves()]) for c in flatents]
    flatents = [(n, ' '.join([l[0] for l in c])) for n, c in flatents]
    candidates = [l[1] for l in flatents]
    return (tokens, tagged, entities, flatents, candidates)

def usefullnlp(s):
    """Uses the fullnlp function"""
    for r in ret:
        r['tokens'], r['tagged'], r['entities'], r['flatents'], r['candidates'] = extractnlp(r['titletext'])
        r['tagged'] = ' '.join(['%s/%s' % (s, tag) for s, tag in r['tagged']])
        r['entities'] = r['entities'].pprint()
        log('Got r: %s' % (r,))
        for c in r['candidates']:
            cands[c] += 1.0 - r['i']/20.0
    scores = sorted([(v, c) for c, v in cands.items()], reverse=1)
    for i, (s, c) in enumerate(scores):
        if c.lower() in x.q.lower():
            c += ' (In Query - IGNORED)'
        scores[i] = (s, c)

def ngramextractor(text, n=(1,2,3,4,5), ret=None, normfunc=None, lweight=0.0):
    """"Extracts n-grams of the given lengths from the given text.
    If the text is a string, then tokenizes it using nltk.
    Passes each token through normfunc, if given.
    If this function returns the empty string, that token is removed.
    If this function returns None, then no n-gram is allowed to span this break.
    Param 'lweight' weights each entry by n**lweight
    Returns (sentences, dict of counts). Sentences are lists of tokens (split by delimiters from normfunc).
    You can optionally pass in 'ret', a dict of existing counts, to append to.
    """
    import nltk
    if ret is None:
        ret = {}
    if isinstance(text, basestring):
        tokens = nltk.word_tokenize(text)
    else:
        tokens = text
    if normfunc:
        tokens = [normfunc(w) for w in tokens if normfunc(w) != '']
        sentences = []
        cur = []
        sentences.append(cur)
        for t in tokens:
            if t is None:
                if cur:
                    cur = []
                    sentences.append(cur)
            else:
                cur.append(t)
    else:
        sentences = [tokens]
    from nkutils import log
    #log('Converted %s to %s to %s' % (text, tokens, sentences))
    for sent in sentences:
        for i, w in enumerate(sent):
            for x in n:
                if i+x > len(sent): continue
                cur = ' '.join(sent[i:i+x])
                #log('  Got i %d, w %s, x %s, cur %s' % (i, w, x, cur))
                if cur not in ret:
                    ret[cur] = 0.0
                ret[cur] += 1*(x**lweight)
    #log('Got ret %s' % (sorted(ret.items(), key=lambda pair: pair[-1], reverse=1),))
    return sentences, ret

def matchstrings(a, b):
    """Matches strings in a very loose way.
    Returns (matches, score), where matches is a list of pairs of matching
    words from a and b, and score is the final normalized similarity score
    (higher is better)."""
    from nkutils import utf, strsim
    import numpy as np
    np.set_printoptions(precision=2, linewidth=200, suppress=1)
    def norm(s):
        """Normalizes a string"""
        s = utf(s).strip().lower()
        return s

    def split(s):
        """Splits a string into components, quite aggressively"""
        import re
        els = re.split('\W+', s)
        els = map(norm, els)
        els = [e for e in els if e and e not in STOP_WORDS]
        return els

    els1 = split(norm(a))
    els2 = split(norm(b))
    #print '%s -> %s' % (a, els1)
    #print '%s -> %s' % (b, els2)
    matches = []
    best = 0.0
    if not els1 or not els2: return (matches, best)
    m = np.zeros((len(els1), len(els2)))
    for i, e1 in enumerate(els1):
        for j, e2 in enumerate(els2):
            c = m[i,j] = strsim(e1, e2)
            #print '  %d,%d = %s vs %s = %s' % (i, j, e1, e2, c)
    while 1:
        n = np.argmax(m)
        i, j = loc = np.unravel_index([n], m.shape)
        s = m[loc]
        #print '  %s, %s, %s' % (n, loc, s)
        if s <= 0: break
        # if we're here, then we want to add this match
        #print 'got %s, %s, %s, %s, %s' % (n, i, j, els1, els2)
        matches.append((els1[i[0]], els2[j[0]]))
        best += s
        m[i,:] = -1
        m[:,j] = -1
        #print '    picking %s, %s' % (matches[-1], best)
        #print m
    if matches:
        best /= float(len(matches))
    return (matches, best)



@memoize
def splitted(s):
    return s.split()

@memoize
def hasWord(term, tag):
    """Returns true if the tag has the term as a subsequence of words."""
    tagwords = splitted(tag)
    termwords = splitted(term)
    nterms = len(termwords)
    term = ' '.join(termwords)
    #print termwords
    #print tagwords
    #print 'nterms:', nterms
    #print term
    #print (term in ' '.join(tagwords))
    # simple check to see if this could possibly contain the word
    if term not in ' '.join(tagwords): return 0
    #log('  Testing term %s (%d terms) with tag %s (%d words)' % (term, nterms, tagwords, len(tagwords)))
    for i in range(len(tagwords)):
        hyptag = ' '.join(tagwords[i:i+nterms])
        #log('    Checking i %d: %s vs %s' % (i, hyptag.encode('utf-8', 'ignore'), term.encode('utf-8', 'ignore')))
        if hyptag == term: return 1
    return 0

@memoize
def wordnetsim(a, b, pos=None):
    """Returns the similarity between the two given terms using wordnet.
    Optionally restricts to a given part of speech, one of 'noun', 'verb', 'adv', 'adj'.
    This searches over all possible pairs of synsets.
    It also takes the max of the wup_similarity and path_similarity score.
    Returns a score between 0 and 1.
    """
    from nltk.corpus import wordnet as wn
    from math import sqrt
    a = wn.morphy(a)
    b = wn.morphy(b)
    if not a or not b: return 0
    posmap = dict(noun=wn.NOUN, verb=wn.VERB, adj=wn.ADJ, adv=wn.ADV)
    alla = wn.synsets(a, pos=posmap.get(pos, None))
    allb = wn.synsets(b, pos=posmap.get(pos, None))
    #score1 = max(a.wup_similarity(b) for a in alla for b in allb)
    score = max(a.path_similarity(b) for a in alla for b in allb)
    if score is None: return 0
    score = sqrt(score)
    #print a, b, score1, score2
    return score

#@memoize
def getTagMatchFunc(matchtype, tags=None, layerkey=None):
    """Returns a function that takes in (queryterm, tag) and returns a score.
    The different matchtypes are:
           'exact': term exactly matches a tag
         'hasword': term is contained within a tag (respecting words)
        'contains': term is contained within a tag
       'unordered': term matches tag (in any order)
          'starts': term starts tag
            'ends': term ends tag
            'typo': term is misspelled version of a tag
            'path': term is close to another (by path similarity in wordnet)
         'synonym': term is a synonym of a tag
        'constant': returns 1
    You can optionally pass in a 'layerkey' as a unique id. This can be used for caches, etc.
    """
    import pickle
    from nkpylib.nknlp import matchstrings
    if matchtype == 'constant': ret = lambda term, tag: 1
    elif matchtype == 'exact': ret = lambda term, tag: 1 if term == tag else 0
    elif matchtype == 'hasword': ret = hasWord
    elif matchtype == 'contains': ret = lambda term, tag: 1 if term and (term in tag or tag in term) else 0
    elif matchtype == 'unordered':
        def ret(term, tag):
            matches, s = matchstrings(term, tag)
            return s if s > 0.3 else 0

    elif matchtype == 'starts': ret = lambda term, tag: 1 if tag.startswith(term) else 0
    elif matchtype == 'ends': ret = lambda term, tag: 1 if tag.endswith(term) else 0
    elif matchtype == 'typo': ret = lambda term, tag: strsim(term, tag) if strsim(term, tag) > 0.7 else 0
    elif matchtype == 'path': ret = lambda term, tag: wordnetsim(term, tag) if wordnetsim(term, tag) > 0.3 else 0
    elif matchtype == 'synonym': ret = lambda term, tag: 0 #TODO fix
    else:
        raise NotImplementedError()
    return (ret)

#@timed
def scoreTermTags(term, tags, matchtypes, layerkey=None):
    """Scores the given term against a list of tags.
    'matchtypes' is a dict mapping different matching types to their weights:
        'constant': returns 1
           'exact': term exactly matches a tag
         'hasword': term is contained within a tag (respecting words)
        'contains': term is contained within a tag
       'unordered': term matches tag (in any order)
          'starts': term starts tag
            'ends': term ends tag
            'typo': term is misspelled version of a tag
            'path': term is close to another (by path similarity in wordnet)
         'synonym': term is a synonym of a tag

    Returns a list of scores, which is the value from the first matching matchtype * the given weight.
    The tags are checked in the order above.
    You can optionally pass in a 'layerkey' as a unique id. This can be used for caches, etc.
    """
    # setup
    ALL_MATCH_TYPES = 'constant exact hasword contains unordered starts ends typo path'.split()
    matchfuncs = dict((mt, getTagMatchFunc(mt, tags=tags, layerkey=layerkey)) for mt in ALL_MATCH_TYPES if mt in matchtypes)
    ret = []
    for tag in tags:
        for mt in ALL_MATCH_TYPES:
            if not tag.strip(): continue # this needs to be continue for the 'else' clause to kick in
            if mt not in matchfuncs: continue # this needs to be continue for the 'else' clause to kick in
            matchfunc = matchfuncs[mt]
            #log('Testing term "%s" vs tag "%s" with mt %s' % (term, tag, mt))
            score = matchfunc(term, tag) * matchtypes[mt]
            if score > 0:
                #print term, tag, mt, matchtypes[mt], score, layerkey
                ret.append(score)
                break
        else:
            ret.append(0)
    assert len(ret) == len(tags)
    return ret

def findall(s, sub):
    """Returns all indices of 'sub' within 's', as a list"""
    ret = []
    cur = 0
    while 1:
        n = s.find(sub, cur)
        if n < 0: return ret
        ret.append(n)
        cur = n+1
    return ret

def cutlabel(s, cuts):
    """Cuts a string s using a set of (n, label) cuts.
    Returns a list of (sub, label) pairs.
    If there was an initial part before the first cut, then it has a label of None.
    If there are no cuts, returns s as a single element, with label None.
    """
    cuts = sorted(cuts)
    # no cuts -> whole string is an element
    if not cuts: return [(s, None)]
    if cuts[0][0] != 0:
        cuts.insert(0, (0, None))
    if cuts[-1][0] < len(s)-1:
        cuts.append((len(s), None))
    locs, labels = zip(*cuts)
    ret = []
    for i, j, label in zip(locs, locs[1:], labels):
        ret.append((s[i:j], label))
    return ret

def nlpnorm(s):
    """Normalizes a token for NLP.
    Currently, this includes:
        - string.lower().strip()
        - if string is delimiter, then return None (forced n-gram break)
        - remove parens, braces, brackets, colons, commas, pipes, etc.
        - remove ...
        - if string contains . inside (but not at end), then force delimiter (assumes this is a website)
    """
    if not s: return ''
    s = s.lower().strip()
    if not isinstance(s, unicode):
        s = unicode(s, 'utf-8', 'ignore')
    for dlm in u'/ @ - | [ ] { } ; : vs. vs at versus « » › â –'.split():
        #log(u'|%s| (%s) vs |%s| (%s)' % (s, type(s), dlm, type(dlm)))
        if s == dlm: return None
    s = u''.join(c for c in s if c not in u"'|()[]{},;`\"«»›–â")
    s = s.replace('...','')
    if '.' in s and not s.endswith('.'): return None
    return s

def inany(el, seq):
    """Returns the first sequence element that el is part of, else None"""
    for item in seq:
        if el in item:
            return item
    return None



if __name__ == '__main__':
    a, b = sys.argv[1:3]
    print matchstrings(a, b)
