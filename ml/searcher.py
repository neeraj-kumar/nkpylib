class RecipeFullSearcher(BaseSearcher):
    """This is a full searcher with a somewhat rigorous input syntax.

    The input query is parsed into clauses, separated by commas. The clauses are "AND"ed together. Each clause is can be one of the following:
    - <field><op><value>: Search for a specific field with a specific value. The operator is one of:
      - =, !=, <, >, <=, >=,
      - in (one of possible values), !in (not in), has (substring), !has (not substring)
      - The field can be any field in the 'dishes' table, but also in linked tables like cuisines or
        comments. In that case, the field name will have a dot in it, like 'Cuisines.Name'.
    - <has or !has><field>: rows which have or don't have a field (regardless of value).
    - <text>: Search for text in all fields, including fuzzy/similarity search.

    At search time, each row is checked against all clauses. If all clauses are satisfied, then the
    row is included in the results.

    For now there is no scoring.
    """
    # note that this is a list because we don't want to match substrings first (like in before !in)
    op_map = [
        ('<=', lambda a, b: a <= b),
        ('>=', lambda a, b: a >= b),
        ('<', lambda a, b: a < b),
        ('>', lambda a, b: a > b),
        ('!=', lambda a, b: a != b),
        ('=', lambda a, b: a == b),
        (' !in ', lambda a, b: a not in b),
        (' in ', lambda a, b: a in b),
        (' !has ', lambda a, b: b not in a),
        (' has ', lambda a, b: b in a),
    ]

    def parse(self, q: str) -> list[tuple[str, str, str]]:
        """Parses the query into clauses."""
        print(f'Full Parsing query {q}')
        # split by commas
        clauses = [c.strip() for c in q.split(',') if c.strip()]
        parsed = []
        for clause in clauses:
            # check for 'has' or 'not has'
            if clause.startswith('has '):
                field = clause[4:]
                op = 'has'
                value = ''
            elif clause.startswith('!has '):
                field = clause[5:]
                op = '!has'
                value = ''
            else:
                # check for <field><op><value>
                for op, _ in self.op_map:
                    if op in clause:
                        field, value = clause.split(op)
                        field = field.strip()
                        # also strip surrounding quotes from value
                        value = value.strip('"\'')
                        #TODO figure out how to do in and not in, since they are lists
                        value = specialize(value.strip())
                        break
                else:
                    # if no operator, then it's a text search
                    field = ''
                    # remove 'text:' from the beginning, and remove surrounding quotes
                    value = clause.strip().replace('text:', '').strip('"\'')
                    op = 'text'
            parsed.append((field.strip(), op, value))
        logger.info(f'Parsed query {q} -> {parsed}')
        return parsed

    def _search(self, q: str, parsed: str, **kw: Any) -> dict[str, object]:
        dishes = self.req.application.dishes
        logger.info(f'Checking {len(dishes)} dishes for {parsed}: {json.dumps(dishes)[:500]}')
        results = []
        def norm(v):
            """Normalize a value for comparison"""
            if isinstance(v, str):
                return v.lower()
            return v

        # accumulate filter results
        for dish in dishes.values():
            filter_values = []
            for field, op, value in parsed:
                cur = False
                if value == '': # check for field existence
                    if op == 'has' and field in dish:
                        cur = True
                    elif op == '!has' and field not in dish:
                        cur = True
                elif op == 'text': # general text search
                    if value.lower() in str(dish).lower():
                        cur = True
                else: # <field><op><value>
                    if field in dish:
                        for f, func in self.op_map:
                            if f == op:
                                to_check = dish[field]
                                if isinstance(to_check, list): #
                                    assert op not in ['=', '!=', '<', '<=', '>', '>='], f'Cannot use {op} with list values'
                                    if 'has' in op:
                                        to_check = '\n'.join([r for r in to_check if r])
                                if func(norm(to_check), norm(value)):
                                    cur = True
                                break
                        else:
                            raise ValueError(f'Unknown operator {op}')
                #print(f'  {field} {op} {value} -> {cur}')
                filter_values.append(cur)
            if all(filter_values):
                results.append(dish['id'])
        s = f'For query {q} -> {parsed}, found {len(results)} results'
        logger.info(s)
        self.add_msg(s)
        return dict(results=results)


class PhotosFullSearcher(MySearcher):
    """Full searcher that requires full (verbose) search query parsing."""
    def parse(self, q: str) -> dict[str, object]:
        """Parses a search query and returns a dict mapping from search type to search params.

        The query is a string with comma separated terms.

        The different types of search:
        - image embeddings [default type]
        - metadata
          - contains key-value pairs separated by =, ==, !=, >, <, >=, <=, in, not in
        - metadata excludes
          - prefix with !
        - text exact match (from "document" in chroma)
          - doc: prefix
        - id regexp search
          - id: prefix, followed by regexp
        - text similarity
          - text: prefix
        """
        q = q.strip()
        orig_q = q
        els = [el.strip() for el in q.split(',') if el.strip()]
        # make sure we don't match a shorter element before a longer one
        md_ops = [
            (' not in ', 'nin'),
            (' nin ', 'nin'),
            (' in ', 'in'),
            ('==', 'eq'),
            ('!=', 'ne'),
            ('>=', 'gte'),
            ('<=', 'lte'),
            ('=', 'eq'),
            ('>', 'gt'),
            ('<', 'lt'),
        ]
        filters = []
        post_filters = []
        image_els = []
        id_filters = []
        doc = []
        place_filters = []

        def make_md_filter(el: str) -> tuple[str, dict]:
            """Converts a string element into a metadata filter, returning a (key, md "where" dict)."""
            for str_op, md_op in md_ops:
                if str_op in el:
                    k, v = el.split(str_op)
                    # special case: convert 'iso_ts' to epoch ts
                    if k == 'iso_ts':
                        k = 'ts'
                        if ' ' in v:
                            v = time.mktime(time.strptime(v, '%Y-%m-%d %H:%M:%S'))
                        else:
                            v = time.mktime(time.strptime(v, '%Y-%m-%d'))
                    return((k, {f'${md_op}': specialize(v)}))


        # go through and accumulate various filters/search types, removing them from the query
        for el in els:
            if el.startswith('pl:'):
                pf = make_md_filter(el[3:])
                if pf:
                    place_filters.append(pf)
                continue
            if el.startswith('doc:'):
                doc.append({'$contains': el[4:]})
                continue
            if el.startswith('!'):
                post_filters.append({'not':el[1:]})
                continue
            if el.startswith('id:'):
                id_filters.append(el[3:])
                continue
            if el.startswith('im:'):
                image_els.append(el[3:])
            md_filter = make_md_filter(el)
            if md_filter:
                filters.append(md_filter)
            else:
                image_els.append(el)
        def join_md_filters(filters):
            """Joins metadata filters into a single dict."""
            if not filters:
                return None
            if len(filters) == 1:
                return {filters[0][0]: filters[0][1]}
            return {'$and': [{f[0]: f[1]} for f in filters]}

        self.log(parser_info=dict(q=q, orig_q=orig_q, els=els))
        ret = dict(image=' '.join(image_els),
                   metadata=join_md_filters(filters),
                   post_filters=post_filters,
                   id_filters=id_filters,
                   place_filters=join_md_filters(place_filters))
        if doc:
            # if there's more than 1, $and them all together
            if len(doc) == 1:
                ret['doc'] = doc[0]
            else:
                ret['doc'] = {'$and': doc}
        self.add_msg(f'Full search: parsed {orig_q} -> {els} ->:\n{json.dumps(ret, indent=2)}.')
        self.log(parsed=ret)
        return ret

    def _search(self, q: str, parsed: object, **kw: Any) -> dict[str, Any]:
        """Runs search with the given query and returns the results.

        This first parses the query to get all the different types of underlying searches to run.
        Then it runs them all (in parallel where possible) and combines the results.
        The results dict includes:
        - 'ids': list of image ids, in the order of relevance
        - 'distances': list of blended distances, in the same order as 'ids'
        - 'types': dict mapping search types to original distances, in the same order as 'ids'
          - if an image id was found in multiple searches, this dict will have multiple entries
        """
        n = kw['n']
        db = self.req.application.db
        kwargs = dict(where=parsed.get('metadata'), include=[])
        post = parsed.get('post_filters', [])
        if post: # if we're doing post filters, we need to include the metadata
            kwargs['include'].append('metadatas')
        if 'doc' in parsed:
            kwargs.update(where_document=parsed['doc'])
        if parsed.setdefault('image', ''):
            kwargs['include'].append('distances')
            kwargs.update(
                n_results=n,
                query_embeddings=[embed_text.single(parsed['image'], model='clip')],
            )
            try:
                ret = self.req.get_col().query(**kwargs)
            except Exception as e:
                logger.error(f'error in query: {e}')
                ret = {}
            # get the first item from each list, since we only searched for 1 item
            ret = {k: ret[k][0] for k in ret if ret[k]}
            ret.setdefault('ids', [])
            # for searches, we can't filter by id in the search, so do it here
            if 'id_filters' in parsed:
                regexps = [re.compile(f) for f in parsed['id_filters']]
                idxes = [i for i, id in enumerate(ret['ids']) if all(re.search(id) for re in regexps)]
                ret = {k: [ret[k][i] for i in idxes] for k in ret}
        else:
            # for 'get', we can filter by id by using the ids from the app
            app = self.req.application
            if 'id_filters' in parsed:
                regexps = [re.compile(f) for f in parsed['id_filters']]
                valid_ids = [id for id in app.get_ids() if all(re.search(id) for re in regexps)]
                kwargs.update(ids=valid_ids)
            kwargs.update(limit=n)
            ret = self.req.get_col().get(**kwargs)
            ret['distances'] = [0.0] * len(ret['ids'])
            ret = {k: ret[k] for k in ret if ret[k] is not None}
        ret.setdefault('ids', [])

        def search_other_col(md_query: dict[str, Any], col_name: str) -> None:
            """Searches another collection for the given metadata query object."""
            kwargs = dict(include=[])
            search_type = 'text'
            # for "contains" queries, embed the text
            and_clause = md_query.get('$and', [])
            if '$contains' in md_query or (and_clause and and_clause[0].get('$contains')):
                kwargs['include'].append('distances')
                kwargs['query_embeddings'] = []
                if '$and' in md_query:
                    for d in md_query['$and']:
                        kwargs['query_embeddings'].append(embed_text.single(d['$contains'], model='sentence'))
                else:
                    kwargs['query_embeddings'].append(embed_text.single(md_query['$contains'], model='sentence'))
                resp = db.get_collection(col_name).query(**kwargs)
            else:
                search_type = 'scored'
                kwargs['include'].append('metadatas')
                kwargs['where'] = md_query
                logger.info(f'Running scored search on {col_name} with {kwargs}')
                _resp = db.get_collection(col_name).get(**kwargs)
                logger.info(f'Got {len(_resp["ids"])} results from {col_name}: {_resp["ids"][:5]}')
                # for scored searches, we need to convert from other ids to image ids
                score_by_id = Counter()
                for md in _resp['metadatas']:
                    # look for 'score:<img id>' fields and take the max score
                    #logger.info(f'checking'
                    for k, v in md.items():
                        if k.startswith('score:'):
                            img_id = k.split(':')[1]
                            score_by_id[img_id] = max(score_by_id[img_id], v)
                logger.info(f'for scored {col_name}: score_by_id: {score_by_id}')
                if not score_by_id:
                    return
                ids, scores = zip(*sorted(score_by_id.items(), key=lambda x: -x[1]))
                resp = {'ids': [list(ids)], 'distances': [[1.0-s for s in scores]]}
                logger.info(f'for scored {col_name} created resp: {resp}')

            # merge the results -- adding new ids where necessary, and then adding distances
            for idx in range(len(resp['ids'])):
                ids, dists = resp['ids'][idx], resp['distances'][idx]
                for id, d in zip(ids, dists):
                    if id in ret['ids']:
                        idx = ret['ids'].index(id)
                        ret['distances'][idx] += d
                    else:
                        ret['ids'].append(id)
                        ret['distances'].append(d)

        # if we had a doc: search, also search the text collection for embedding-based results
        if parsed.get('doc'):
            search_other_col(md_query=parsed['doc'], col_name=TEXT_COL_NAME)
        # if we had places, accumulate those results
        if parsed.get('place_filters'):
            search_other_col(md_query=parsed['place_filters'], col_name=PLACES_COL_NAME)
        if not ret['ids']:
            self.add_msg('No results found.')
            return ret
        # sort the results by distance
        ret = {k: [ret[k][i] for i in np.argsort(ret['distances'])] for k in ret}
        # apply post filters
        idxes = np.ones(len(ret['ids']), dtype=bool)
        for p in post:
            if 'not' in p: # remove results where the given metadata field name is not present
                to_remove = [idx for idx, md in enumerate(ret['metadatas']) if p['not'] in md]
                idxes[to_remove] = False
        ret = {k: [ret[k][i] for i in range(len(ret['ids'])) if idxes[i]] for k in ret}
        # add types
        ret['types'] = []
        for idx in range(len(ret['ids'])):
            cur = {t: ret['distances'][idx] if t == 'image' else 0 for t in parsed}
            ret['types'].append(cur)
        return ret

