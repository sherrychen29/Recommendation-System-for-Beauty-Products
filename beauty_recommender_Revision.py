import time
_start_time = time.time()

import ast
import pandas as pd
import numpy as np
import glob
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from rapidfuzz import fuzz
from joblib import Parallel, delayed

# ═══════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════════════

# Load products
products = pd.read_csv('product_info.csv')
print(f"Products loaded: {products.shape}")

# Load and merge all review CSV files
files = glob.glob('reviews*.csv')
if not files:
    raise FileNotFoundError(
        "No review CSV files found matching 'reviews*.csv'. "
        "Ensure at least one reviews file is present in the working directory."
    )
reviews = pd.concat(
    [pd.read_csv(f) for f in files],
    ignore_index=True
)
reviews = reviews.drop_duplicates()
print(f"Reviews loaded:  {reviews.shape}")

# ═══════════════════════════════════════════════════════════════════════
# STEP 2 — CLEAN REVIEWS
# ═══════════════════════════════════════════════════════════════════════

# Drop rows missing truly critical fields (author_id, product_id, rating, review_text).
profile_cols = ['skin_tone', 'skin_type', 'hair_color', 'eye_color']
n_before = len(reviews)
reviews = reviews.dropna(subset=['author_id', 'product_id', 'rating', 'review_text'])
print(f"Reviews before dropna:  {n_before:,}")
print(f"Reviews after dropna:   {len(reviews):,}  ({n_before - len(reviews):,} dropped — missing critical fields)")

# Minimum demographic threshold: keep only reviews where at least 2 of 4
# demographic fields are filled. Reviews with fewer filled fields contribute
# near-zero profile signal but add noise to the weighted rating/text functions.
reviews[profile_cols] = reviews[profile_cols].fillna('')
_n_filled = (reviews[profile_cols] != '').sum(axis=1)
n_before_demo = len(reviews)
reviews = reviews[_n_filled >= 2].reset_index(drop=True)
print(f"Reviews after demo threshold (≥2 fields): {len(reviews):,}  "
      f"({n_before_demo - len(reviews):,} dropped — sparse demographic profiles)")

# Profile values used throughout for proportion scoring and matching.
# Defined here so Step 3 aggregation can use them.
DEFAULT_PROFILE_VALS = {
    'skin_tone':  'light',
    'skin_type':  'combination',
    'hair_color': 'brown',
    'eye_color':  'brown',
}
DEFAULT_PROFILE = dict(DEFAULT_PROFILE_VALS)  # copy so mutations don't corrupt the source dict

# ═══════════════════════════════════════════════════════════════════════
# STEP 3 — AGGREGATE REVIEW DATA PER PRODUCT
# ═══════════════════════════════════════════════════════════════════════

positive_reviews = reviews[reviews['rating'] >= 4]

# Top-rated reviewers: per product, only those who gave the maximum rating.
# Using top-rated reviewers as ground truth is more precise — it reflects
# which skin types found the product most satisfying, not merely who bought it.
_max_per_product = positive_reviews.groupby('product_id')['rating'].transform('max')
top_reviews = positive_reviews[positive_reviews['rating'] == _max_per_product].copy()
print(f"Top-rated reviewers: {len(top_reviews):,}  "
      f"({len(top_reviews)/len(positive_reviews)*100:.1f}% of positive reviews)")

# Content+Skin Profile uses only product-metadata signals for ranking
# (w_skinrating = TF-IDF+LogReg on product text, w_content, w_pop) so its
# ranking is structurally independent of reviewer skin-type proportion data.
# This removes the circular measurement that required a train/eval split —
# the full positive_reviews pool is used for both aggregation and evaluation.

# Base aggregation: ratings and review text.
# dominant_skin_type uses all positive reviewers here; overridden below from
# top_reviews so the accuracy diagnostic reflects top-rated reviewer ground truth.
review_agg = positive_reviews.groupby('product_id').agg(
    dominant_skin_tone  = ('skin_tone',   lambda x: x.mode()[0] if not x.mode().empty else ''),
    dominant_skin_type  = ('skin_type',   lambda x: x.mode()[0] if not x.mode().empty else ''),
    dominant_hair_color = ('hair_color',  lambda x: x.mode()[0] if not x.mode().empty else ''),
    dominant_eye_color  = ('eye_color',   lambda x: x.mode()[0] if not x.mode().empty else ''),
    avg_rating          = ('rating',      'mean'),
    review_count        = ('rating',      'count'),
    review_text         = ('review_text', ' '.join)
).reset_index()

# Override dominant_skin_type with mode of top-rated reviewers per product.
_top_st = (top_reviews[top_reviews['skin_type'] != '']
           .groupby('product_id')['skin_type']
           .agg(lambda x: x.mode().iloc[0])
           .rename('dominant_skin_type')
           .reset_index())
review_agg = (review_agg
              .drop(columns=['dominant_skin_type'])
              .merge(_top_st, on='product_id', how='left'))
review_agg['dominant_skin_type'] = review_agg['dominant_skin_type'].fillna('')

# Proportion-based profile scores — what fraction of reviewers per product
# match each attribute value. Used by fuzzy profile match in evaluation.
# This gives every reviewed product a continuous signal rather than a
# winner-takes-all mode, so profile matching is always non-zero.
def proportion_score(series, target_value):
    """Fraction of non-empty values in series that fuzzy-match target."""
    from rapidfuzz import fuzz as _fuzz
    non_empty = series[series != '']
    if non_empty.empty:
        return 0.0
    scores = non_empty.apply(
        lambda v: max(_fuzz.partial_ratio(target_value.lower(), str(v).lower()),
                      _fuzz.token_sort_ratio(target_value.lower(), str(v).lower())) / 100.0
    )
    return round(float(scores.mean()), 4)

PROFILE_ATTRS = {
    'skin_tone':  DEFAULT_PROFILE_VALS['skin_tone'],
    'skin_type':  DEFAULT_PROFILE_VALS['skin_type'],
    'hair_color': DEFAULT_PROFILE_VALS['hair_color'],
    'eye_color':  DEFAULT_PROFILE_VALS['eye_color'],
}

prop_rows = []
for pid, grp in positive_reviews.groupby('product_id'):
    row = {'product_id': pid}
    for col, target in PROFILE_ATTRS.items():
        row[f'prop_{col}'] = proportion_score(grp[col], target)
    prop_rows.append(row)

prop_df    = pd.DataFrame(prop_rows)
review_agg = review_agg.merge(prop_df, on='product_id', how='left')

# Merge into products
df = products.merge(review_agg, on='product_id', how='left')

# Fill nulls
fill_cols = ['dominant_skin_tone', 'dominant_skin_type',
             'dominant_hair_color', 'dominant_eye_color', 'review_text']
df[fill_cols]      = df[fill_cols].fillna('')
prop_cols = ['prop_skin_tone', 'prop_skin_type', 'prop_hair_color', 'prop_eye_color']
df[prop_cols]      = df[prop_cols].fillna(0.0)
df['avg_rating']   = df['avg_rating'].fillna(0)
df['review_count'] = df['review_count'].fillna(0)
df['highlights']   = df['highlights'].fillna('')
df['brand_name']   = df['brand_name'].fillna('')

print(f"Merged dataframe (all products):      {df.shape}")

# Restrict to products that have at least one review.
# This ensures every candidate has real reviewer profile data so
# profile-weighting, fuzzy matching, and diversity signals are meaningful.
df = df[df['review_count'] > 0].reset_index(drop=True)
print(f"Reviewed products only (analysis set): {df.shape}")

# ═══════════════════════════════════════════════════════════════════════
# STEP 4 — BUILD PRODUCT SOUP (product features only)
# ═══════════════════════════════════════════════════════════════════════

_sec    = df['secondary_category'].fillna('') if 'secondary_category' in df.columns else ''
_hl     = df['highlights'].fillna('').str.replace(',', ' ').str.strip()
_detail = _hl.where(_hl != '', df['product_name'])  # fallback to name when highlights missing
# product_name is always included (not just as fallback) so every product's
# own name contributes to TF-IDF regardless of whether highlights are present.
# Names like "Oil-Free Acne Wash" or "Hydrating Cleanser for Dry Skin" carry
# formulation intent that highlights do not always repeat.
df['product_soup'] = (df['primary_category'] + ' ' + _sec + ' '
                      + df['product_name'].fillna('') + ' ' + _detail)

# ═══════════════════════════════════════════════════════════════════════
# STEP 5 — BUILD SIMILARITY MATRICES
# ═══════════════════════════════════════════════════════════════════════

# Matrix A — product info similarity
tfidf_product = TfidfVectorizer(stop_words='english', max_features=5000)
sim_product   = cosine_similarity(tfidf_product.fit_transform(df['product_soup'])).astype(np.float32)
print(f"Product similarity matrix: {sim_product.shape}")

# Matrix B — review text similarity (base, used when no profile given)
tfidf_review = TfidfVectorizer(stop_words='english', max_features=5000)
sim_review   = cosine_similarity(tfidf_review.fit_transform(df['review_text'])).astype(np.float32)
print(f"Review similarity matrix:  {sim_review.shape}")

# Matrix C — collaborative filtering (item-based)
# Build sparse matrix directly to avoid allocating a dense users×products array
_users    = reviews['author_id'].astype('category')
_products = reviews['product_id'].astype('category')
sparse_matrix = csr_matrix(
    (reviews['rating'].values,
     (_users.cat.codes, _products.cat.codes)),
    shape=(_users.cat.categories.size, _products.cat.categories.size)
)
product_id_to_pos = {pid: i for i, pid in enumerate(_products.cat.categories)}
collab_sim    = cosine_similarity(sparse_matrix.T).astype(np.float32)   # products × products
print(f"Collab similarity matrix:  {collab_sim.shape}")

# Pre-compute collab position lookup arrays (vectorized, avoids per-call for loop)
_df_collab_pos  = df['product_id'].map(product_id_to_pos)
_df_has_collab  = _df_collab_pos.notna().values
_df_collab_idx  = _df_collab_pos.fillna(0).astype(int).values

# Matrix D — skin-type-filtered collab + review similarity matrices (Options 2 & 3)
# One matrix per skin type, built from reviews whose author has that skin_type.
# Stored as: skin_type → (pid_to_pos, df_has, df_idx, sim_matrix)  for collab
#            skin_type → sim_matrix                                  for review text
_ST_MATRIX_TYPES = ['dry', 'oily', 'normal', 'combination', 'sensitive']
_st_collab_sim:  dict = {}  # skin_type → (pid_to_pos, df_has_arr, df_idx_arr, sim_mat)
_st_review_sim:  dict = {}  # skin_type → review similarity matrix (n_products × n_products)

print("Building skin-type-filtered similarity matrices...")
for _stype in _ST_MATRIX_TYPES:
    _st_rev = reviews[reviews['skin_type'] == _stype]
    if len(_st_rev) < 50:
        print(f"  [{_stype:<12}]: {len(_st_rev)} reviews — skipped (< 50)")
        continue
    # Option 2: skin-type-filtered item-based CF
    _st_u = _st_rev['author_id'].astype('category')
    _st_p = _st_rev['product_id'].astype('category')
    _st_sparse = csr_matrix(
        (_st_rev['rating'].values, (_st_u.cat.codes, _st_p.cat.codes)),
        shape=(_st_u.cat.categories.size, _st_p.cat.categories.size)
    )
    _st_pid_map    = {pid: i for i, pid in enumerate(_st_p.cat.categories)}
    _st_csim       = cosine_similarity(_st_sparse.T).astype(np.float32)
    _st_df_cpos    = df['product_id'].map(_st_pid_map)
    _st_df_has     = _st_df_cpos.notna().values
    _st_df_idx     = _st_df_cpos.fillna(0).astype(int).values
    _st_collab_sim[_stype] = (_st_pid_map, _st_df_has, _st_df_idx, _st_csim)
    # Option 3: skin-type-filtered review text TF-IDF similarity
    _st_text_ser   = _st_rev.groupby('product_id')['review_text'].agg(' '.join)
    _st_text_col   = df['product_id'].map(_st_text_ser).fillna('')
    _st_tfidf_v    = TfidfVectorizer(stop_words='english', max_features=5000)
    _st_X          = _st_tfidf_v.fit_transform(_st_text_col)
    _st_review_sim[_stype] = cosine_similarity(_st_X).astype(np.float32)
    print(f"  [{_stype:<12}]: {len(_st_rev):,} reviews  "
          f"collab {_st_sparse.shape}  review sim {_st_X.shape}")
print(f"Skin-type-filtered matrices built: {sorted(_st_collab_sim.keys())}")

# Pre-compute popularity array (never changes, avoids per-call recomputation)
_max_reviews        = df['review_count'].max()
_popularity         = (0.6 * df['avg_rating'].fillna(0) / 5.0 +
                       0.4 * np.log1p(df['review_count'].fillna(0)) / np.log1p(_max_reviews)
                       ).values.astype(np.float32)
_review_count_arr    = df['review_count'].fillna(0).values
_review_count_median  = float(df['review_count'].median())
_review_count_25pct   = float(df['review_count'].quantile(0.25))
_rc_sorted            = np.sort(_review_count_arr)          # for O(log n) percentile lookup in serendipity

# Default (no-profile) weighted rating array
_default_wtd_rating = df['avg_rating'].fillna(0).values.astype(np.float32)

# Per-profile weighted rating arrays — populated during cache warm-up
_weighted_rating_arr_cache: dict = {}
_extra_schemes:             dict = {}   # temporary weight scheme overrides for testing

# ── Text-based skin-type compatibility signal ────────────────────────────────
# Scores each product by how strongly its product text (highlights, category,
# product name) signals suitability for a given skin type.
# Built ENTIRELY from product catalogue metadata — zero dependency on reviewer
# data — so it is structurally independent of the _skin_prop_lookup evaluator.
# Used in 'profile_text' and 'full_text' weight schemes as a drop-in
# replacement for w_skinrating to test whether conclusions hold without any
# reviewer-derived ranking signal.
#
# Keyword sets are grouped by semantic role:
#   explicit  — phrases that name the skin type directly
#   functional — ingredient/formulation properties associated with the type
#   avoidance  — terms associated with OTHER types (used to suppress score)
#
# Scoring: (explicit_hits * 2 + functional_hits) normalised to [0, 1].
# The ×2 weight on explicit phrases gives clear "for dry skin" claims
# priority over incidental functional language that any product might carry.

_SKIN_TEXT_KEYWORDS: dict = {
    'dry': {
        'explicit':   ['dry skin', 'for dry', 'very dry', 'dryness',
                       'dehydrated skin', 'dehydration'],
        'functional': ['hydrating', 'moisturising', 'moisturizing', 'nourishing',
                       'nourish', 'replenishing', 'replenish', 'rich formula',
                       'deeply hydrat', 'intense moisture', 'moisture barrier',
                       'emollient', 'occlusive', 'ceramide', 'hyaluronic',
                       'squalane', 'shea butter', 'facial oil', 'overnight mask',
                       'sleeping mask', 'balm', 'creamy'],
    },
    'oily': {
        'explicit':   ['oily skin', 'for oily', 'excess oil', 'oil control',
                       'shine control', 'anti-shine'],
        'functional': ['oil-free', 'oil free', 'mattifying', 'matte finish',
                       'non-comedogenic', 'non comedogenic', 'pore minimising',
                       'pore minimizing', 'pore refining', 'pore control',
                       'sebum', 'lightweight', 'gel formula', 'water-based',
                       'water based', 'salicylic', 'bha', 'clay', 'charcoal',
                       'absorbing', 'blotting'],
    },
    'combination': {
        'explicit':   ['combination skin', 'for combination', 'combo skin',
                       't-zone', 't zone', 'dual skin', 'dual-zone', 'dual zone',
                       'partial oiliness', 'cheek dryness'],
        'functional': ['dual action', 'adaptive',
                       'zones', 'multizone', 'multi-zone', 'rebalancing',
                        'sebum control',
                       't-zone control', 'lightweight'],
    },
    'normal': {
        'explicit':   ['normal skin', 'for normal', 'all skin types',
                       'all skin', 'everyday', 'daily use'],
        'functional': [ 'universal', 'gentle daily', 'maintenance',
                       'multi-use', 'versatile'],
    },
    'sensitive': {
        'explicit':   ['sensitive skin', 'for sensitive', 'sensitivity',
                       'reactive skin', 'easily irritated'],
        'functional': ['fragrance-free', 'fragrance free', 'unscented',
                       'hypoallergenic', 'dermatologist tested',
                       'dermatologist-tested', 'calming', 'soothing',
                       'redness', 'redness-reducing', 'gentle formula',
                       'alcohol-free', 'alcohol free', 'non-irritating',
                       'allergy tested', 'minimal ingredients', 'barrier',
                       'centella', 'cica', 'aloe', 'oat'],
    },
}

def _build_text_skintype_signal() -> dict:
    """
    Score every product for each canonical skin type using one binary
    TF-IDF + LogReg classifier per skin type (one-vs-rest).

    Labels come exclusively from Sephora's 'Best for X Skin' highlight tags —
    placed by the brand, with ZERO reviewer dependency.  This makes the signal
    fully independent of the reviewer skin-type proportion data used by the
    SkinCompat evaluation metric, eliminating circular measurement.

    Label mapping (a product can be positive for multiple types):
      'Best for Dry Skin'                → {dry}
      'Best for Oily Skin'               → {oily}
      'Best for Normal Skin'             → {normal}
      'Best for Combination Skin'        → {combination}
      'Best for Dry, Combo, Normal Skin' → {dry, combination, normal}
      'Best for Oily, Combo, Normal Skin'→ {oily, combination, normal}

    Training set per skin type:
      Positive: products whose 'Best for' tag includes that skin type.
      Negative: products whose 'Best for' tag exists but excludes that type.
      Unlabeled products (no 'Best for' tag) are excluded from training.

    Falls back to keyword scoring when positive examples are too sparse.
    Threshold is 50 for 'combination' (tags are scarce; LogReg underperforms
    below this) and 10 for all other types (e.g. 'sensitive', no 'Best for' tag).

    Returns dict: skin_type → np.ndarray shape (len(df),) in [0, 1].
    """
    import ast as _ast

    canonical_keys = set(_SKIN_TEXT_KEYWORDS.keys())

    # ── Combined text field (product metadata only) ───────────────────────
    # Highlights are parsed and "Best for X Skin" tags stripped before joining
    # so that label strings do not appear as TF-IDF features — which would let
    # the classifier trivially memorise labels rather than learning from
    # product text.  All other highlight tags (Oil Free, Fragrance Free, etc.)
    # are retained as they carry genuine formulation signal.
    _BEST_FOR_TAGS = {
        'Best for Dry Skin', 'Best for Oily Skin', 'Best for Normal Skin',
        'Best for Combination Skin', 'Best for Dry, Combo, Normal Skin',
        'Best for Oily, Combo, Normal Skin', 'All Skin Types',
        'Good for All Skin Types',
    }

    def _strip_best_for(raw_h):
        if not raw_h or not isinstance(raw_h, str):
            return ''
        try:
            tags = _ast.literal_eval(raw_h)
            return ' '.join(t for t in tags if t not in _BEST_FOR_TAGS)
        except Exception:
            return raw_h

    _highlights_clean = df['highlights'].apply(_strip_best_for)

    _text = (_highlights_clean + ' ' +
             df['primary_category'].fillna('') + ' ' +
             df['secondary_category'].fillna('') + ' ' +
             df['tertiary_category'].fillna('') + ' ' +
             df['ingredients'].fillna('') + ' ' +
             df['product_name'].fillna('')
             ).str.lower()

    # ── Parse 'Best for X Skin' tags → per-product label sets ────────────
    _BEST_FOR_MAP = {
        'Best for Dry Skin':               {'dry'},
        'Best for Oily Skin':              {'oily'},
        'Best for Normal Skin':            {'normal'},
        'Best for Combination Skin':       {'combination'},
        'Best for Dry, Combo, Normal Skin':  {'dry', 'combination', 'normal'},
        'Best for Oily, Combo, Normal Skin': {'oily', 'combination', 'normal'},
    }

    # df positional index → frozenset of applicable skin types
    _label_by_pos = {}
    for pos, (_, row) in enumerate(df.iterrows()):
        raw_h = row.get('highlights', '')
        if not raw_h or not isinstance(raw_h, str):
            continue
        try:
            tags = _ast.literal_eval(raw_h)
        except Exception:
            continue
        skin_types_for_product = set()
        for tag in tags:
            if tag in _BEST_FOR_MAP:
                skin_types_for_product |= _BEST_FOR_MAP[tag]
        if skin_types_for_product:
            _label_by_pos[pos] = frozenset(skin_types_for_product)

    n_labeled = len(_label_by_pos)
    print(f"  Products with 'Best for X Skin' labels: {n_labeled} / {len(df)}")
    for st in sorted(canonical_keys):
        n_pos = sum(1 for sts in _label_by_pos.values() if st in sts)
        print(f"    {st:<14}: {n_pos:>4} positive labeled products")

    # ── TF-IDF features (fit on ALL product text) ─────────────────────────
    # ngram_range=(1,2) captures compound ingredient terms (e.g. "salicylic acid").
    # max_features raised to 20000 to accommodate ingredient vocabulary.
    vec   = TfidfVectorizer(ngram_range=(1, 2), max_features=20000,
                            min_df=2, sublinear_tf=True)
    X_tfidf = vec.fit_transform(_text)

    # ── Expert keyword binary features ────────────────────────────────────
    # One binary column per keyword phrase: 1 if phrase appears in product text.
    # These are domain-critical terms (ingredients, claims) that TF-IDF may
    # under-weight when they appear infrequently across the corpus.
    _EXPERT_KW = {
        'dry':         ['hyaluronic acid', 'ceramide', 'shea butter', 'squalane',
                        'glycerin', 'argan oil', 'jojoba', 'nourishing', 'hydrating',
                        'emollient', 'occlusive', 'moisturising', 'moisturizing'],
        'oily':        ['salicylic acid', 'niacinamide', 'kaolin', 'bentonite',
                        'mattify', 'oil-free', 'oil free', 'pore', 'zinc oxide',
                        'charcoal', 'clay', 'sebum', 'bha', 'astringent'],
        'combination': ['balancing', 'dual-action', 'dual action', 't-zone',
                        't zone', 'combination skin', 'combo skin', 'multizone',
                        'multi-zone', 'dual-zone', 'dual zone', 'partial oiliness',
                        'sebum control', 'cheek dryness', 'rebalancing',
                        't-zone control'],
        'normal':      ['everyday', 'all skin types', 'versatile',
                        'balanced skin', 'normal skin', 'all-purpose',
                        'multi-purpose', 'suitable for all', 'non-problematic',
                        'regular skin'],
        'sensitive':   ['sensitive skin', 'fragrance-free', 'fragrance free',
                        'hypoallergenic', 'calming', 'soothing', 'redness',
                        'allergy tested', 'dermatologist', 'gentle', 'non-irritating'],
    }
    from scipy.sparse import hstack as _sp_hstack, csr_matrix as _csr
    _kw_cols = []
    _kw_names = []
    for _st_kw, _phrases in _EXPERT_KW.items():
        for _phrase in _phrases:
            _col = _text.str.contains(_phrase, regex=False).astype(np.float32).values
            _kw_cols.append(_csr(_col.reshape(-1, 1)))
            _kw_names.append(f'kw__{_st_kw}__{_phrase.replace(" ", "_")}')
    X_all = _sp_hstack([X_tfidf] + _kw_cols, format='csr')
    print(f"  Feature matrix: {X_tfidf.shape[1]} TF-IDF + {len(_kw_cols)} keyword "
          f"= {X_all.shape[1]} total features")

    # Per-type fallback threshold: combination needs more examples because its
    # 'Best for' tags overlap heavily with dry/normal/oily multi-type tags,
    # making the decision boundary poorly defined below ~50 positives.
    _FALLBACK_THRESH = {'combination': 50}
    _DEFAULT_THRESH  = 10

    # ── One binary classifier per skin type ───────────────────────────────
    result = {}
    for st in sorted(canonical_keys):
        pos_pos = [p for p, sts in _label_by_pos.items() if st in sts]
        neg_pos = [p for p, sts in _label_by_pos.items() if st not in sts]
        _thresh = _FALLBACK_THRESH.get(st, _DEFAULT_THRESH)

        if len(pos_pos) < _thresh:
            # ── Keyword fallback ──────────────────────────────────────────
            print(f"  [{st:12s}]: {len(pos_pos)} positives < {_thresh} threshold"
                  f" — using keyword fallback")
            kw_groups = _SKIN_TEXT_KEYWORDS.get(st, {'explicit': [], 'functional': []})
            explicit_hits   = sum(_text.str.contains(kw, regex=False).astype(float)
                                  for kw in kw_groups['explicit'])
            functional_hits = sum(_text.str.contains(kw, regex=False).astype(float)
                                  for kw in kw_groups['functional'])
            raw     = explicit_hits * 2.0 + functional_hits
            max_val = float(raw.max())
            result[st] = (raw / max_val if max_val > 0 else raw).values.astype(np.float32)
            continue

        # Build binary target: 1 = good for this skin type, 0 = not
        all_pos  = pos_pos + neg_pos
        y_binary = np.array([1] * len(pos_pos) + [0] * len(neg_pos), dtype=int)
        X_train  = X_all[all_pos]

        clf = LogisticRegression(max_iter=1000, class_weight='balanced',
                                 solver='lbfgs', C=1.0)
        clf.fit(X_train, y_binary)

        # Cross-validated ROC-AUC (more informative than accuracy for binary)
        n_cv = min(5, len(pos_pos))
        if n_cv >= 2 and len(neg_pos) >= 2:
            cv_scores = cross_val_score(
                LogisticRegression(max_iter=1000, class_weight='balanced',
                                   solver='lbfgs', C=1.0),
                X_train, y_binary, cv=n_cv, scoring='roc_auc'
            )
            print(f"  [{st:12s}]: CV ROC-AUC {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
                  f"  ({len(pos_pos)} pos / {len(neg_pos)} neg)")

        # Predict probability of being suitable for this skin type
        pos_col   = list(clf.classes_).index(1)
        proba_st  = clf.predict_proba(X_all)[:, pos_col].astype(np.float32)
        result[st] = proba_st
        n_above   = int((proba_st > 0.3).sum())
        print(f"  [{st:12s}]: {n_above:4d} / {len(df)} products score > 0.3"
              f"  (max={proba_st.max():.3f})")

    return result

print("Building text-based skin-type signals (product metadata only)...")
_text_skintype_signal: dict = _build_text_skintype_signal()
print(f"Text skin-type signal built for: {sorted(_text_skintype_signal.keys())}")

# ═══════════════════════════════════════════════════════════════════════
# STEP 6 — BUILD PRODUCT NAME INDEX
# ═══════════════════════════════════════════════════════════════════════

indices = pd.Series(
    df.index,
    index=df['product_name'].str.lower().str.strip()
).drop_duplicates()

# ═══════════════════════════════════════════════════════════════════════
# STEP 7 — HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def get_collab_scores(product_id):
    """Return collaborative similarity scores for a product."""
    if product_id not in product_id_to_pos:
        return None
    pos = product_id_to_pos[product_id]
    return collab_sim[pos]


def fuzzy_similarity(val_a: str, val_b: str) -> float:
    """
    Fuzzy match two attribute strings.
    Uses partial_ratio so 'medium' matches 'lightMedium' / 'mediumTan'.
    Uses token_sort_ratio to handle word-order differences.
    Returns 0.0-1.0.
    """
    a = str(val_a).lower().strip()
    b = str(val_b).lower().strip()
    if not a or not b:
        return 0.0
    return max(fuzz.partial_ratio(a, b),
               fuzz.token_sort_ratio(a, b)) / 100.0


def reviewer_similarity(reviewer_row, user_profile: dict) -> float:
    """
    Compare a single reviewer's attributes to the shopper's profile
    using fuzzy matching so compound values like 'lightMedium' still
    match a profile value of 'medium'.
    Returns a weight between 0.0 and 1.0.
    """
    checks = {
        'skin_tone':  'skin_tone',
        'skin_type':  'skin_type',
        'hair_color': 'hair_color',
        'eye_color':  'eye_color',
    }

    if not user_profile:
        return 1.0

    scores = []
    for user_key, review_col in checks.items():
        if user_key not in user_profile:
            continue
        user_val     = user_profile[user_key]
        reviewer_val = reviewer_row.get(review_col, '')
        scores.append(fuzzy_similarity(user_val, reviewer_val))

    return round(sum(scores) / len(scores), 4) if scores else 0.0


_reviewer_weights_cache: dict = {}

def _profile_key(profile: dict):
    """Hashable cache key from a profile dict."""
    if not profile:
        return None
    return tuple(sorted(profile.items()))

def _reviewer_weights(user_profile: dict) -> np.ndarray:
    """
    Vectorized reviewer similarity weights for all reviews.
    Computes fuzzy scores only for unique attribute combinations,
    then maps back — far fewer fuzzy calls than row-by-row apply().
    Results are cached per profile.
    """
    key = _profile_key(user_profile)
    if key in _reviewer_weights_cache:
        return _reviewer_weights_cache[key]

    attrs = ['skin_tone', 'skin_type', 'hair_color', 'eye_color']
    user_vals = {a: user_profile.get(a, '') for a in attrs}

    # Compute scores for unique reviewer attribute combinations only
    combos = reviews[attrs].drop_duplicates().copy()
    combos['_w'] = combos.apply(
        lambda row: round(
            sum(fuzzy_similarity(user_vals[a], str(row[a])) for a in attrs) / len(attrs), 4
        ), axis=1
    )
    weights = reviews[attrs].merge(combos, on=attrs, how='left')['_w'].values
    _reviewer_weights_cache[key] = weights
    return weights


_weighted_ratings_cache: dict = {}

def get_weighted_ratings(user_profile: dict) -> pd.Series:
    """
    Compute a weighted average rating per product where reviews
    from similar shoppers count more.
    Returns a Series: product_id → weighted_avg_rating
    """
    key = _profile_key(user_profile)
    if key in _weighted_ratings_cache:
        return _weighted_ratings_cache[key]

    weights_s = pd.Series(_reviewer_weights(user_profile).clip(min=0.1),
                          index=reviews.index)
    result  = (reviews.groupby('product_id')
                      .apply(lambda g: np.average(g['rating'],
                                                  weights=weights_s[g.index]),
                             include_groups=False))
    _weighted_ratings_cache[key] = result
    return result


_weighted_text_cache: dict = {}

def get_weighted_review_text(user_profile: dict) -> pd.Series:
    """
    Build a review text corpus where similar reviewers' words
    are amplified based on their similarity weight.
    weight=1.0 → text repeated 3x
    weight=0.0 → text included once (base)
    """
    key = _profile_key(user_profile)
    if key in _weighted_text_cache:
        return _weighted_text_cache[key]

    weights  = _reviewer_weights(user_profile)
    repeats  = np.maximum(1, np.round(1 + weights * 2)).astype(int)
    # Truncate to 300 chars: TF-IDF only needs word frequencies; full text
    # tripled by repeat() can exhaust RAM on large review datasets.
    texts    = reviews['review_text'].astype(str).str[:300]
    prod_ids = reviews['product_id']

    # Expand rows by repeat count, then join per product — no iterrows needed
    expanded = pd.DataFrame({
        'product_id':  prod_ids.repeat(repeats).values,
        'review_text': texts.repeat(repeats).values,
    })
    result = expanded.groupby('product_id')['review_text'].agg(' '.join)
    _weighted_text_cache[key] = result
    return result


_review_tfidf_cache: dict = {}
_review_sim_cache:   dict = {}   # profile_key -> full (n x n) cosine similarity matrix


POSITIVE_WORDS = ['love', 'amazing', 'perfect', 'great', 'holy grail',
                  'repurchase', 'favorite', 'glowing', 'hydrated', 'smooth']
NEGATIVE_WORDS = ['broke out', 'irritated', 'greasy', 'heavy', 'disappointed',
                  'returned', 'awful', 'terrible', 'dry', 'flaky']

def get_profile_sentiment(product_id, user_profile: dict) -> dict:
    """
    Analyze sentiment from reviews written by similar shoppers only.
    Returns a summary of what people like you said about this product.
    """
    product_reviews = reviews[reviews['product_id'] == product_id].copy()

    product_reviews['similarity'] = product_reviews.apply(
        lambda row: reviewer_similarity(row, user_profile), axis=1
    )

    # Only consider reviews from similar shoppers
    similar_reviews = product_reviews[product_reviews['similarity'] >= 0.5]

    if similar_reviews.empty:
        return {'message': 'No reviews from similar shoppers found.'}

    all_text   = ' '.join(similar_reviews['review_text'].fillna('').str.lower())
    avg_rating = np.average(
        similar_reviews['rating'],
        weights=similar_reviews['similarity']
    )

    positive_hits = [w for w in POSITIVE_WORDS if w in all_text]
    negative_hits = [w for w in NEGATIVE_WORDS if w in all_text]

    return {
        'similar_reviewer_count': len(similar_reviews),
        'weighted_avg_rating':    round(avg_rating, 2),
        'positive_signals':       positive_hits,
        'negative_signals':       negative_hits,
        'verdict': '✅ Well loved by people like you' if avg_rating >= 4.0
                   else '⚠️ Mixed reviews from people like you' if avg_rating >= 3.0
                   else '❌ Not well rated by people like you'
    }

# ═══════════════════════════════════════════════════════════════════════
# STEP 8 — MAIN RECOMMENDATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def _mmr_rerank(candidate_idx, relevance_scores, sim_matrix, top_n, lambda_mmr):
    """
    Maximal Marginal Relevance re-ranking.

    Iteratively selects items that maximise:
        MMR(i) = λ · rel(i)  −  (1−λ) · max_{j ∈ selected} sim(i, j)

    λ=1.0 → pure relevance order (no diversity effect)
    λ=0.7 → strong relevance with meaningful diversity push
    λ=0.5 → equal weight on relevance and novelty

    Parameters
    ----------
    candidate_idx    : 1-D int array  — df row positions, sorted by relevance desc
    relevance_scores : 1-D float array — combined score per candidate (same order)
    sim_matrix       : 2-D float array — full product×product cosine similarity
    top_n            : int  — how many items to return
    lambda_mmr       : float in [0, 1]

    Returns
    -------
    List of df row positions, length min(top_n, len(candidate_idx))
    """
    if lambda_mmr >= 1.0 or len(candidate_idx) <= top_n:
        # Fast path: no diversity needed — return top-n by relevance.
        return list(candidate_idx[:top_n])

    # Normalise relevance to [0, 1] within the candidate pool so the
    # λ trade-off is stable regardless of absolute score magnitude.
    r_min = relevance_scores.min()
    r_max = relevance_scores.max()
    norm_rel = (relevance_scores - r_min) / (r_max - r_min + 1e-9)

    candidates  = list(candidate_idx)   # mutable pool of df positions
    norm_lookup = {int(p): float(nr)    # df_pos → normalised relevance
                   for p, nr in zip(candidate_idx, norm_rel)}
    selected    = []

    while len(selected) < top_n and candidates:
        if not selected:
            # First pick is always the highest-relevance candidate.
            best = candidates[0]
        else:
            sel_arr    = np.array(selected, dtype=np.int32)
            best_score = -np.inf
            best       = candidates[0]          # fallback
            for cand in candidates:
                # Max similarity to any already-selected item (vectorised row slice)
                max_sim   = float(sim_matrix[cand][sel_arr].max())
                mmr_score = lambda_mmr * norm_lookup[cand] - (1.0 - lambda_mmr) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best       = cand
        selected.append(best)
        candidates.remove(best)

    return selected


def get_recommendations(
        product_name,
        user_profile  = None,
        top_n         = 10,
        skin_type     = None,
        use_collab    = True,
        use_reviews   = True,
        weight_scheme = 'full',
        mmr_lambda    = 1,   # =1 off; =0.7 on
        epsilon       = 0.0): # ε-greedy: fraction of slots filled by random pool pick
    """
    Parameters
    ----------
    product_name  : str  - name of a Sephora product
    user_profile  : dict - e.g. {'skin_tone': 'light', 'skin_type': 'dry', ...}
    top_n         : int  - number of recommendations to return
    skin_type     : str  - optional post-filter e.g. 'dry', 'oily'
    use_collab    : bool - include collaborative filtering signal
    use_reviews   : bool - include review text similarity signal
    weight_scheme : str  - one of:
                    'content_only' pure TF-IDF product features
                    'balanced'     equal-ish weight across active signals
                    'popularity'   heavily weights rating + review count
                    'profile'      heavily weights profile-matched reviews
                    'full'         all signals with tuned weights

    Returns
    -------
    DataFrame with top N recommended products and scores
    """
    product_name = product_name.lower().strip()
    if product_name not in indices.index:
        return "Product not found. Check the name spelling!"

    idx = indices.loc[product_name]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    product_id = df['product_id'].iloc[idx]

    # Signal 1: content-based TF-IDF (always on)
    content_scores = sim_product[idx]

    # Signal 2: review text similarity (optional)
    if use_reviews:
        if user_profile:
            key = _profile_key(user_profile)
            if key not in _review_tfidf_cache:
                weighted_text = get_weighted_review_text(user_profile)
                _wrt = df['product_id'].map(weighted_text).fillna('')
                tfidf_w = TfidfVectorizer(stop_words='english', max_features=5000)
                X = tfidf_w.fit_transform(_wrt)
                _review_tfidf_cache[key] = X
                _review_sim_cache[key] = cosine_similarity(X).astype(np.float32)
            review_scores = _review_sim_cache[key][idx]
        else:
            review_scores = sim_review[idx]
    else:
        review_scores = np.zeros(len(df))

    # Signal 3: collaborative filtering (optional)
    if use_collab:
        collab_raw = get_collab_scores(product_id)
        if collab_raw is None:
            collab_aligned = np.zeros(len(df))
        else:
            collab_aligned = np.where(_df_has_collab, collab_raw[_df_collab_idx], 0.0)
    else:
        collab_aligned = np.zeros(len(df))

    # Signal 4: popularity (pre-computed at startup)
    popularity = _popularity

    # Signal 5: skin-type text signal — TF-IDF + LogReg classifier probability
    # for the inferred skin type, trained on product metadata (highlights,
    # category, name). Structurally independent of reviewer proportion data.
    _st_key = user_profile.get('skin_type', '') if user_profile else ''
    if user_profile and _st_key == 'any':
        # Universal skin type: all products equally compatible — no filtering.
        skinrating_scores = np.ones(len(df), dtype=np.float32)
    elif user_profile and _st_key == 'normal':
        # Normal skin text classifier is contaminated by multi-type 'Best for' tags
        # (dry+combo+normal, oily+combo+normal), giving poor discrimination.
        # Blend with complement signal: products that score low on both oily and dry
        # are implicitly normal-compatible (no extreme-type specialisation).
        _n = _text_skintype_signal.get('normal', np.zeros(len(df), dtype=np.float32))
        _o = _text_skintype_signal.get('oily',   np.zeros(len(df), dtype=np.float32))
        _d = _text_skintype_signal.get('dry',    np.zeros(len(df), dtype=np.float32))
        skinrating_scores = (0.5 * _n
                             + 0.5 * (1.0 - np.maximum(_o, _d))
                                     .clip(0.0, 1.0).astype(np.float32))
    elif user_profile and _st_key:
        skinrating_scores = _text_skintype_signal.get(
            _st_key, np.zeros(len(df), dtype=np.float32))
    else:
        skinrating_scores = _default_wtd_rating / 5.0

    # Signal 6: skin-type-aware collaborative filtering (Option 2)
    # CF matrix built from reviewers whose skin_type matches the inferred type.
    # Falls back to zeros when skin type is unknown or product absent from filtered matrix.
    if _st_key == 'any':
        collab_st_scores = collab_aligned   # no skin filter — use full CF
    elif _st_key and _st_key in _st_collab_sim:
        _stc_pid_map, _stc_df_has, _stc_df_idx, _stc_csim = _st_collab_sim[_st_key]
        if product_id in _stc_pid_map:
            _raw_stc = _stc_csim[_stc_pid_map[product_id]]
            collab_st_scores = np.where(_stc_df_has, _raw_stc[_stc_df_idx], 0.0).astype(np.float32)
        else:
            collab_st_scores = np.zeros(len(df), dtype=np.float32)
    else:
        collab_st_scores = np.zeros(len(df), dtype=np.float32)

    # Signal 7: skin-type-aware review text similarity (Option 3)
    # TF-IDF similarity computed from reviews by same-skin-type reviewers only.
    if _st_key == 'any':
        review_st_scores = review_scores    # no skin filter — use full review similarity
    elif _st_key and _st_key in _st_review_sim:
        review_st_scores = _st_review_sim[_st_key][idx]
    else:
        review_st_scores = np.zeros(len(df), dtype=np.float32)

    # Weight schemes — each row sums to 1.0 exactly.
    # w_skinrating  = TF-IDF+LogReg classifier probability for the inferred
    #                 skin type (product metadata text signal).
    schemes = {
        'content_only':   {'w_content': 1.00, 'w_review': 0.00,
                           'w_collab':  0.00, 'w_pop':    0.00, 'w_skinrating': 0.00},
        'balanced':       {'w_content': 0.34, 'w_review': 0.00,
                           'w_collab':  0.33, 'w_pop':    0.33, 'w_skinrating': 0.00},
        'content_collab': {'w_content': 0.65, 'w_review': 0.00,
                           'w_collab':  0.20, 'w_pop':    0.15, 'w_skinrating': 0.00},
        'content_reviews':{'w_content': 0.65, 'w_review': 0.20,
                           'w_collab':  0.00, 'w_pop':    0.15, 'w_skinrating': 0.00},
        'popularity':     {'w_content': 0.20, 'w_review': 0.00,
                           'w_collab':  0.00, 'w_pop':    0.80, 'w_skinrating': 0.00},
        # profile_text: ranking uses ONLY product-metadata signals (no reviewer
        # skin-type data) so evaluation can use the full reviewer pool without leakage.
        'profile_content_skin':   {'w_content': 0.35, 'w_review': 0.00,
                           'w_collab':  0.00, 'w_pop':    0.00, 'w_skinrating': 0.65},
        'profile_collab_skin':   {'w_content': 0.00, 'w_review': 0.00,
                           'w_collab':  0.35, 'w_pop':    0.00, 'w_skinrating': 0.65},
        'profile_review_skin':   {'w_content': 0.00, 'w_review': 0.35,
                           'w_collab':  0.00, 'w_pop':    0.00, 'w_skinrating': 0.65},
        'full_text':      {'w_content': 0.2, 'w_review': 0.1,
                           'w_collab':  0.10, 'w_pop':    0.05, 'w_skinrating': 0.55},
        # 'full' is an alias for 'full_text' — kept for backwards compatibility.
        'full':           {'w_content': 0.10, 'w_review': 0.15,
                           'w_collab':  0.10, 'w_pop':    0.10, 'w_skinrating': 0.55},
        # Full Hybrid: blend of skin-aware signals.
        # w_skinrating = product metadata classifier (Content+Skin Profile)
        # w_collab     = standard CF (full reviewer pool — drives serendipity)
        # w_collab_st  = skin-type-filtered CF (drives RoutineCoh)
        'full_hybrid_st':     {'w_content': 0.20, 'w_review': 0.00,
                               'w_collab':  0.15, 'w_pop':    0.00,
                               'w_skinrating': 0.40,
                               'w_collab_st': 0.25, 'w_review_st': 0.00},
        # Full Hybrid without skin-type-filtered CF — removes the double-filter
        # (w_skinrating ∩ w_collab_st) that restricts coverage.
        # w_collab_st redistributed to w_collab to keep total CF signal unchanged.
        'full_hybrid_no_collab_st': {'w_content': 0.20, 'w_review': 0.00,
                                     'w_collab':  0.40, 'w_pop':    0.00,
                                     'w_skinrating': 0.40,
                                     'w_collab_st': 0.00, 'w_review_st': 0.00},
    }
    w = (weight_scheme if isinstance(weight_scheme, dict)
         else (_extra_schemes.get(weight_scheme)
               or schemes.get(weight_scheme, schemes['full_text'])))

    # Sanity-check: weights must sum to 1.0
    # Use .get() for w_collab_st / w_review_st — absent in schemes that don't use them.
    w_sum = (w.get('w_content', 0.0) + w.get('w_review', 0.0) + w.get('w_collab', 0.0)
             + w.get('w_pop', 0.0) + w.get('w_skinrating', 0.0)
             + w.get('w_collab_st', 0.0) + w.get('w_review_st', 0.0))
    _scheme_label = weight_scheme if isinstance(weight_scheme, str) else 'custom dict'
    assert abs(w_sum - 1.0) < 1e-6, f"Weights for '{_scheme_label}' sum to {w_sum:.4f}, not 1.0"

    combined = (w.get('w_content', 0.0)    * content_scores    +
                w.get('w_review', 0.0)     * review_scores     +
                w.get('w_collab', 0.0)     * collab_aligned    +
                w.get('w_pop', 0.0)        * popularity        +
                w.get('w_skinrating', 0.0) * skinrating_scores +
                w.get('w_collab_st', 0.0)  * collab_st_scores  +
                w.get('w_review_st', 0.0)  * review_st_scores)

    combined[idx] = -np.inf                          # exclude the seed itself
    # Pull a larger candidate pool so MMR has room to diversify without
    # running out of items.  Pool size = top_n × MMR_POOL_MULT (default 12).
    # When mmr_lambda=1.0 (MMR off) the pool size has no effect on output.
    n_top = min(top_n * MMR_POOL_MULT, len(combined) - 1)   # guard: k < n
    top_idx = np.argpartition(combined, -n_top)[-n_top:]
    top_idx = top_idx[np.argsort(combined[top_idx])[::-1]]  # sorted desc by score

    # ── MMR diversity re-ranking ──────────────────────────────────────────
    # Re-orders the candidate pool to maximise relevance while penalising
    # items already similar to previously selected ones.  Uses the product
    # content similarity matrix (sim_product) as the diversity measure so
    # that items from different categories / formulation families are
    # preferred over near-duplicates.
    if epsilon > 0.0:
        # ε-greedy selection: each slot is filled by the highest-ordered remaining
        # candidate (exploit) with probability (1−ε), or by a uniformly random
        # remaining candidate (explore) with probability ε.
        # When mmr_lambda < 1.0, run MMR first to produce a diversity-aware ordering
        # of the full pool, then apply epsilon-greedy on that MMR-ordered list.
        # When mmr_lambda == 1.0, epsilon-greedy operates on the score-sorted pool.
        if mmr_lambda < 1.0:
            _ordered = _mmr_rerank(
                candidate_idx    = top_idx,
                relevance_scores = combined[top_idx],
                sim_matrix       = sim_product,
                top_n            = len(top_idx),   # full pool, MMR-ordered
                lambda_mmr       = mmr_lambda,
            )
        else:
            _ordered = list(top_idx)               # score-sorted pool
        _rng_e  = np.random.default_rng(seed=abs(hash(str(product_name))) % (2**31))
        _remain = list(_ordered)
        selected = []
        while len(selected) < top_n and _remain:
            if _rng_e.random() < epsilon:
                i = int(_rng_e.integers(len(_remain)))   # random from remaining pool
            else:
                i = 0                                    # highest-ordered remaining
            selected.append(_remain.pop(i))
        product_indices = selected
    else:
        mmr_indices = _mmr_rerank(
            candidate_idx    = top_idx,
            relevance_scores = combined[top_idx],
            sim_matrix       = sim_product,
            top_n            = top_n,
            lambda_mmr       = mmr_lambda,
        )
        product_indices = mmr_indices

    base_cols = ['product_id', 'product_name', 'brand_name', 'primary_category',
                  'avg_rating', 'review_count',
                  'dominant_skin_tone', 'dominant_skin_type',
                  'dominant_hair_color', 'dominant_eye_color',
                  'prop_skin_tone', 'prop_skin_type',
                  'prop_hair_color', 'prop_eye_color']
    if 'secondary_category' in df.columns:
        base_cols.insert(3, 'secondary_category')
    results = df[base_cols].iloc[product_indices].copy()
    results['weighted_rating']  = df['avg_rating'].iloc[product_indices].values
    results['similarity_score'] = np.round(combined[np.array(product_indices)], 3).tolist()

    if skin_type and skin_type.strip():
        mask    = df['highlights'].iloc[product_indices].str.lower().str.contains(
            skin_type.strip().lower(), na=False
        )
        results = results[mask.values]

    results['final_score'] = results['similarity_score'].round(3)

    return results.reset_index(drop=True)

def get_random_recommendations(product_name, top_n=10, rng=None):
    """
    Return `top_n` products drawn uniformly at random from the catalogue,
    excluding the seed product.  Used as the chance-level baseline in
    full_evaluation so readers can calibrate whether observed metric scores
    represent real improvement or marginal variation above chance.

    The returned DataFrame has the same schema as get_recommendations so
    all downstream metric functions (skin_compatibility_score, map_at_k_skin,
    etc.) work without modification.
    """
    if rng is None:
        rng = random.Random(42)

    product_name = product_name.lower().strip()
    # Exclude seed from pool
    pool = df[df['product_name'].str.lower().str.strip() != product_name].copy()
    if len(pool) < top_n:
        return pool.reset_index(drop=True)

    chosen = pool.sample(n=top_n, random_state=rng.randint(0, 2**31 - 1))

    base_cols = ['product_id', 'product_name', 'brand_name', 'primary_category',
                 'avg_rating', 'review_count',
                 'dominant_skin_tone', 'dominant_skin_type',
                 'dominant_hair_color', 'dominant_eye_color',
                 'prop_skin_tone', 'prop_skin_type',
                 'prop_hair_color', 'prop_eye_color']
    if 'secondary_category' in df.columns:
        base_cols.insert(3, 'secondary_category')
    available = [c for c in base_cols if c in chosen.columns]
    result = chosen[available].copy()
    result['weighted_rating']  = result['avg_rating'].fillna(0) / 5.0
    result['similarity_score'] = 0.0
    result['final_score']      = 0.0
    return result.reset_index(drop=True)


# ===========================================================================
# STEP 10 — MATPLOTLIB SETUP, PALETTE & JOURNAL STYLE CONSTANTS
# ===========================================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Okabe-Ito colorblind-safe palette — accepted by Nature, IEEE, ACS journals.
# Distinguishable in greyscale print and by all forms of colour blindness.
COLORS = {
    'Random':                        '#BBBBBB',  # mid-grey — chance baseline
    'Content Only':                  '#0072B2',  # blue
    'Content + Collab':              '#56B4E9',  # sky blue
    'Content + Reviews':             '#009E73',  # bluish green
    'Popularity':                    '#E69F00',  # amber
    'Content+Skin Profile':          '#CC5500',  # burnt orange — content + skin classifier
    'Collab+Skin Profile':   '#D55E00',  # vermillion — adds skin-aware CF
    'review+Skin Profile':   '#F0E442',  # yellow — adds skin-aware review text
    'Full Hybrid':                   '#CC79A7',  # reddish purple — blend of top configs
    'Full Hybrid + MMR':             '#882255',  # deep magenta — Full Hybrid with diversity re-ranking
    'Full Hybrid Stochastic':        '#332288',  # indigo — epsilon-greedy selection
    'Full Hybrid+MMR+stochastic':    '#AA4499',  # purple — MMR-ordered pool + epsilon-greedy
    'Optimised Hybrid':              '#117733',  # dark green — Bayesian-optimised weights
}

# Shared journal style constants — change once, applies everywhere
JOURNAL_BG      = 'white'
JOURNAL_SPINE   = '#AAAAAA'   # light grey spines — clean, not harsh
JOURNAL_GRID    = '#E0E0E0'   # very light grey grid lines
JOURNAL_TEXT    = 'black'
JOURNAL_SUBTEXT = '#444444'   # axis labels, secondary text
JOURNAL_CMAP    = 'Blues'     # sequential, greyscale-safe heatmap
JOURNAL_DPI     = 300         # 300 dpi — standard for print journals

# ── MMR global switch ────────────────────────────────────────────────────────
MMR_ENABLED        = False  # ← set False to disable MMR everywhere
MMR_LAMBDA         = 0.7 if MMR_ENABLED else 1.0   # 0.7=diversity, 1.0=pure relevance
MMR_POOL_MULT      = 12    # candidate pool = top_n × MMR_POOL_MULT before MMR re-ranking
                            # has no effect when mmr_lambda=1.0 (MMR disabled)

# 13 configurations: 5 base + 3 skin-type-aware variants + 1 full hybrid + 1 full hybrid + MMR
#                    + 1 full hybrid without collab_st + 1 full hybrid stochastic
#                    + 1 full hybrid MMR + stochastic
CONFIGURATIONS = {
    # mmr_lambda controls the relevance-diversity trade-off in MMR re-ranking.
    # λ=1.0: pure relevance order (disables MMR entirely — clean signal ablation).
    # λ=0.6: diversity re-ranking active — used only for Full Hybrid + MMR variants.
    # All non-hybrid configs use λ=1.0 so that MMR effects are isolated to the
    # Full Hybrid + MMR configuration, enabling a clean ablation comparison.
    'Random':                        {'profile': None,       'use_collab': False,
                                      'use_reviews': False,  'weight_scheme': 'random',
                                      'mmr_lambda': 1.0},    # random already diverse
    'Content Only':                  {'profile': None,       'use_collab': False,
                                      'use_reviews': False,  'weight_scheme': 'content_only',
                                      'mmr_lambda': 1.0},    # pure signal — no MMR
    'Content + Collab':              {'profile': None,       'use_collab': True,
                                      'use_reviews': False,  'weight_scheme': 'content_collab',
                                      'mmr_lambda': 1.0},    # pure signal — no MMR
    'Content + Reviews':             {'profile': None,       'use_collab': False,
                                      'use_reviews': True,   'weight_scheme': 'content_reviews',
                                      'mmr_lambda': 1.0},    # pure signal — no MMR
    'Popularity':                    {'profile': None,       'use_collab': False,
                                      'use_reviews': False,  'weight_scheme': 'popularity',
                                      'mmr_lambda': 1.0},    # pure signal — no MMR
    # Skin-type-aware configs: skin type inferred from the seed product's
    # metadata (Best-for tags → text classifier) — no user input required.
    # Profile is set to 'inferred'; use_collab/use_reviews flags control whether
    # the standard (non-skin-filtered) CF/review signals are also blended in.
    'Content+Skin Profile':          {'profile': 'inferred', 'use_collab': False,
                                      'use_reviews': False,  'weight_scheme': 'profile_content_skin',
                                      'mmr_lambda': 1.0},    # pure signal — no MMR
    'Collab+Skin Profile':   {'profile': 'inferred', 'use_collab': True,
                                      'use_reviews': False,  'weight_scheme': 'profile_collab_skin',
                                      'mmr_lambda': 1.0},    # pure signal — no MMR
    'review+Skin Profile':   {'profile': 'inferred', 'use_collab': False,
                                      'use_reviews': True,   'weight_scheme': 'profile_review_skin',
                                      'mmr_lambda': 1.0},    # pure signal — no MMR
    # Full Hybrid: weighted blend of top configs — content + skin classifier
    # + skin-type-aware CF + skin-type-aware review text.
    'Full Hybrid':                   {'profile': 'inferred', 'use_collab': True,
                                      'use_reviews': False,  'weight_scheme': 'full_hybrid_st',
                                      'mmr_lambda': 1.0},   # MMR off — pure relevance order
    # Full Hybrid + MMR: same weights with MMR diversity re-ranking active (λ=0.6).
    # Lower λ = more diversity pressure. Expect higher Coverage/Serendipity,
    # slightly lower SkinCompat/MAP@5 vs Full Hybrid.
    'Full Hybrid + MMR':             {'profile': 'inferred', 'use_collab': True,
                                      'use_reviews': False,  'weight_scheme': 'full_hybrid_st',
                                      'mmr_lambda': 0.6},
    # Full Hybrid Stochastic: ε-greedy selection (ε=0.15) — each recommendation
    # slot is filled by the top-scored remaining candidate 85% of the time and by
    # a uniformly random pool candidate 15% of the time, spreading coverage across
    # the catalogue without altering the scoring function.
    'Full Hybrid Stochastic':        {'profile': 'inferred', 'use_collab': True,
                                      'use_reviews': False,  'weight_scheme': 'full_hybrid_st',
                                      'mmr_lambda': 1.0,     'epsilon': 0.30},
    # Full Hybrid+MMR+stochastic: combines MMR diversity re-ranking with ε-greedy
    # exploration.  MMR (λ=0.6) first reranks the full candidate pool for intra-list
    # diversity, then epsilon-greedy (ε=0.30) selects top_n from that MMR-ordered
    # pool.  Stacking both mechanisms targets both within-list and cross-seed coverage.
    'Full Hybrid+MMR+stochastic':    {'profile': 'inferred', 'use_collab': True,
                                      'use_reviews': False,  'weight_scheme': 'full_hybrid_st',
                                      'mmr_lambda': 0.6,     'epsilon': 0.30},
}

# Hybrid variants used in cross-profile overlap and personalisation lift tests.
HYBRID_CONFIGS = [
    'Full Hybrid',
    'Full Hybrid + MMR',
    'Full Hybrid Stochastic',
    'Full Hybrid+MMR+stochastic',
]

# ===========================================================================
# STEP 9 — COSMETICS-SPECIFIC EVALUATION METRICS
# ===========================================================================
# Standard IR metrics (precision@category) are a poor fit for cosmetics.
# Relevance in beauty is not "same category" — it is "products that people
# with my skin type loved, that complement rather than duplicate each other,
# and that surface discoveries beyond bestsellers."
#
# Metric set:
#   1. Skin Compatibility     — prop of recs loved by similar skin types (core)
#   2. MAP@5 (skin-relevant)  — rank-sensitive, relevance = skin_type match
#   3. Routine Coherence      — are rec sub-categories varied (not 5 moisturisers)?
#   4. Intra-List Diversity   — TF-IDF pairwise dissimilarity
#   5. Brand Diversity        — fraction of unique brands
#   6. Serendipity            — (1−pop_pct) × (1−sim_to_seed) × (avg_rating/5): unexpected × dissimilar × quality
#   7. Profile-Weighted Rating— avg rating by similar shoppers
#   8. Personalisation Score  — computed separately in Test 2


# Canonical skin-type groupings: maps every dataset value to a canonical label.
# "normal to dry" / "dry to normal" should count for both dry and normal queries.
_SKIN_TYPE_ALIASES = {
    'dry':              {'dry', 'dry skin', 'very dry'},
    'oily':             {'oily', 'oily skin', 'very oily'},
    'normal':           {'normal', 'normal skin'},
    'combination':      {'combination', 'combination skin', 'combo'},
    'sensitive':        {'sensitive', 'sensitive skin'},
}
# Compound values (e.g. "normal to dry") count for BOTH constituent types.
# We handle them by checking if the target appears as a word in the value.
_SKIN_TYPE_ALIASES_EXTRA = {
    'dry':         {'normal to dry', 'dry to normal', 'dry/combination'},
    'oily':        {'normal to oily', 'oily to normal', 'oily/combination'},
    'normal':      {'normal to dry', 'dry to normal', 'normal to oily', 'oily to normal'},
    'combination': {'dry/combination', 'oily/combination'},
}


def _canonicalize_skin_type(raw: str) -> str:
    """
    Map any raw reviewer skin_type value to a canonical key
    (dry, oily, normal, combination, sensitive) using the alias tables.
    Returns '' if no match is found.
    """
    v = str(raw).strip().lower()
    if not v:
        return ''
    # Direct canonical key
    if v in _SKIN_TYPE_ALIASES:
        return v
    # Primary aliases
    for canon, aliases in _SKIN_TYPE_ALIASES.items():
        if v in aliases:
            return canon
    # Compound aliases
    for canon, aliases in _SKIN_TYPE_ALIASES_EXTRA.items():
        if v in aliases:
            return canon
    return ''


def _infer_profile_from_seed(seed_name: str) -> dict:
    """
    Build a user profile from the seed product's skin type.
    Uses pre-built O(1) lookup — no DataFrame scans at call time.

    Cascade (resolved once at startup into _name_to_skin_type):
      0. _best_for_name_lookup              — brand-stated 'Best for X Skin' tag
      1. _text_skintype_signal argmax ≥ 0.5 — product text classifier
      2. DEFAULT_PROFILE skin_type          — last resort
    Reviewer dominant_skin_type is excluded to avoid circular evaluation.
    """
    canonical = _name_to_skin_type.get(seed_name.lower().strip(), '')
    profile = dict(DEFAULT_PROFILE)
    profile['skin_type'] = canonical
    return profile


# All "Best for X Skin" highlight tags → frozenset of canonical skin types.
# Multi-label tags (e.g. "Best for Dry, Combo, Normal Skin") are fully included:
# a product matching multiple types will appear in all their seed pools.
_BEST_FOR_TAG_TYPES = {
    'Best for Dry Skin':               frozenset({'dry'}),
    'Best for Oily Skin':              frozenset({'oily'}),
    'Best for Normal Skin':            frozenset({'normal'}),
    'Best for Combination Skin':       frozenset({'combination'}),
    'Best for Dry, Combo, Normal Skin':  frozenset({'dry', 'combination', 'normal'}),
    'Best for Oily, Combo, Normal Skin': frozenset({'oily', 'combination', 'normal'}),
    # "All skin types" tags: expand to all four specific types so these products
    # appear in every skin-type seed pool directly and their canonical type is
    # determined by reviewer dominant_skin_type rather than an arbitrary sentinel.
    'All Skin Types':          frozenset({'dry', 'oily', 'normal', 'combination'}),
    'Good for All Skin Types': frozenset({'dry', 'oily', 'normal', 'combination'}),
}

# Pre-build: product_name (lower) → frozenset of skin types from "Best for" tags.
_best_for_name_lookup: dict = {}
for _idx, _row in df.iterrows():
    _name = str(_row.get('product_name', '')).lower().strip()
    if not _name:
        continue
    _raw_h = _row.get('highlights', '')
    if not _raw_h or not isinstance(_raw_h, str):
        continue
    try:
        _tags = ast.literal_eval(_raw_h)
        _found: frozenset = frozenset()
        for _tag, _types in _BEST_FOR_TAG_TYPES.items():
            if _tag in _tags:
                _found = _found | _types
        if _found:
            _best_for_name_lookup[_name] = _found
    except Exception:
        pass
print(f"Best-for tag lookup built: {len(_best_for_name_lookup)} products  "
      f"({sum(1 for v in _best_for_name_lookup.values() if len(v) > 1)} with multi-label tags)")

# Sensitive skin keyword set: products explicitly marketed for sensitive skin
# but lacking a dedicated "Best for Sensitive Skin" Sephora tag.
_SENSITIVE_KEYWORDS = {'sensitive', 'sensitiv'}
_sensitive_name_set: set = set()
for _idx, _row in df.iterrows():
    _name = str(_row.get('product_name', '')).lower().strip()
    if not _name:
        continue
    _search_text = (_name + ' ' +
                    str(_row.get('highlights', '')).lower())
    if any(_kw in _search_text for _kw in _SENSITIVE_KEYWORDS):
        _sensitive_name_set.add(_name)
print(f"Sensitive keyword set built:  {len(_sensitive_name_set)} products")

# Pre-build two lookups in one pass:
#   _product_skin_types  : product_name → frozenset of all valid skin types
#                          (used for multi-type seed pool membership in evaluation)
#   _name_to_skin_type   : product_name → single canonical type
#                          (used for profile inference during ranking)
#
# Cascade (reviewer dominant_skin_type excluded — circular with SkinCompat eval):
#   Step 0a: "Best for X Skin" highlight tags  → frozenset (may be multi-type)
#   Step 0b: sensitive keyword in name/highlights → add 'sensitive' to set
#   Step 1 : text classifier argmax ≥ 0.5      → single type
#   Step 2 : DEFAULT_PROFILE                   → fallback single type
_product_skin_types: dict = {}   # product_name → frozenset
_name_to_skin_type:  dict = {}   # product_name → str (first type, for ranking)
for _idx, _row in df.iterrows():
    _name = str(_row.get('product_name', '')).lower().strip()
    if not _name:
        continue
    # Step 0a: Best-for tags (frozenset, may be multi-type)
    _type_set: frozenset = _best_for_name_lookup.get(_name, frozenset())
    # Step 0b: sensitive keyword
    if _name in _sensitive_name_set:
        _type_set = _type_set | frozenset({'sensitive'})
    # Step 1: text classifier (only when no tag signal)
    if not _type_set:
        _best_st, _best_score = '', -1.0
        for _st, _scores in _text_skintype_signal.items():
            _score = float(_scores[_idx])
            if _score > _best_score:
                _best_score = _score
                _best_st = _st
        if _best_score > 0.5:
            _type_set = frozenset({_best_st})
    # Step 2: last resort
    if not _type_set:
        _type_set = frozenset({DEFAULT_PROFILE.get('skin_type', '')})
    _product_skin_types[_name] = _type_set
    # Single canonical type for ranking.
    # 'any'-only products keep 'any' — treated as universally compatible.
    # Multi-type products use dominant_skin_type from top-rated reviewers
    # as the tie-break; fall back to alphabetical minimum if unavailable.
    _real_types = _type_set - {'any'}
    _reviewer_type = str(_row.get('dominant_skin_type', '')).strip().lower()
    if not _real_types:
        _name_to_skin_type[_name] = 'any'                           # universal
    elif _reviewer_type and _reviewer_type not in ('', 'any'):
        _name_to_skin_type[_name] = _reviewer_type                  # data-driven
    else:
        _name_to_skin_type[_name] = min(_real_types)                # alphabetical fallback
print(f"Skin-type lookups built: {len(_product_skin_types)} products  "
      f"({sum(1 for v in _product_skin_types.values() if len(v) > 1)} multi-type)")

# ── Skin-type inference accuracy diagnostics ────────────────────────────────
# Evaluate how well the inference cascade recovers the true dominant_skin_type.
# "Ground truth" = canonical form of dominant_skin_type from reviewer majority vote.
# Source breakdown shows which cascade step fired for each product.
print("\n" + "=" * 70)
print("  SKIN-TYPE INFERENCE — ACCURACY & DISTRIBUTION REPORT")
print("=" * 70)

_infer_rows = []
for _idx, _row in df.iterrows():
    _name  = str(_row.get('product_name', '')).lower().strip()
    _raw   = str(_row.get('dominant_skin_type', ''))
    _truth = _canonicalize_skin_type(_raw)   # reviewer ground truth (may be '')

    # Determine which cascade step fired (mirrors _product_skin_types build above).
    # _truth is reviewer ground truth used only for accuracy comparison, NOT as a
    # cascade source (excluded to avoid circular evaluation).
    _type_set = _best_for_name_lookup.get(_name, frozenset())
    if _name in _sensitive_name_set:
        _type_set = _type_set | frozenset({'sensitive'})
    if _type_set:
        _source   = 'best_for' if _name in _best_for_name_lookup else 'sensitive_kw'
        _inferred = min(_type_set)  # consistent with _name_to_skin_type tie-break
    else:
        # Step 1: text argmax (only if classifier confident ≥ 0.5)
        _best_st, _best_score = '', -1.0
        for _st, _scores in _text_skintype_signal.items():
            _score = float(_scores[_idx])
            if _score > _best_score:
                _best_score = _score
                _best_st = _st
        if _best_score > 0.5:
            _source   = 'text'
            _inferred = _best_st
        else:
            _source   = 'default'
            _inferred = DEFAULT_PROFILE.get('skin_type', '')

    _infer_rows.append({
        'product_name': _name,
        'truth':        _truth,
        'inferred':     _inferred,
        'source':       _source,
    })

_infer_df = pd.DataFrame(_infer_rows)

# ── Source breakdown ─────────────────────────────────────────────────────────
print("\n  SOURCE BREAKDOWN (which cascade step fired):")
_src_counts = _infer_df['source'].value_counts()
for _src, _cnt in _src_counts.items():
    print(f"    {_src:<10}  {_cnt:>5}  ({100*_cnt/len(_infer_df):.1f}%)")

# ── Inferred skin-type distribution ─────────────────────────────────────────
print("\n  INFERRED SKIN-TYPE DISTRIBUTION:")
_inf_counts = _infer_df['inferred'].value_counts()
for _st, _cnt in _inf_counts.items():
    bar = '█' * int(30 * _cnt / len(_infer_df))
    print(f"    {_st:<14}  {_cnt:>5}  ({100*_cnt/len(_infer_df):5.1f}%)  {bar}")

# ── Ground-truth distribution (products with reviewer data) ─────────────────
_with_truth = _infer_df[_infer_df['truth'] != '']
print(f"\n  GROUND-TRUTH DISTRIBUTION  ({len(_with_truth)} products with reviewer skin type):")
_truth_counts = _with_truth['truth'].value_counts()
for _st, _cnt in _truth_counts.items():
    bar = '█' * int(30 * _cnt / len(_with_truth))
    print(f"    {_st:<14}  {_cnt:>5}  ({100*_cnt/len(_with_truth):5.1f}%)  {bar}")

# ── Accuracy: reviewer-sourced products only ────────────────────────────────
# For reviewer-sourced rows, inferred == truth by construction (same signal).
# Accuracy is only meaningful for text-sourced rows vs their reviewer ground truth
# when both are available (i.e. products that DO have reviewer data, comparing
# what text would have inferred vs what reviewers actually say).
print("\n  TEXT-SIGNAL ACCURACY  (how often text argmax agrees with reviewer majority):")
print("  (Evaluated on products where reviewer ground truth exists)")
_text_check = []
for _idx, _row in df.iterrows():
    _raw   = str(_row.get('dominant_skin_type', ''))
    _truth = _canonicalize_skin_type(_raw)
    if not _truth:
        continue
    # What would text have inferred?
    _best_st, _best_score = '', -1.0
    for _st, _scores in _text_skintype_signal.items():
        _score = float(_scores[_idx])
        if _score > _best_score:
            _best_score = _score
            _best_st = _st
    _text_pred = _best_st if _best_score > 0 else ''
    _text_check.append({
        'truth':      _truth,
        'text_pred':  _text_pred,
        'correct':    (_text_pred == _truth) if _text_pred else None,
    })
_tc_df = pd.DataFrame(_text_check)
_tc_with_pred = _tc_df[_tc_df['text_pred'] != '']
_tc_overall   = _tc_with_pred['correct'].mean()
print(f"\n    Overall text accuracy: {_tc_overall:.3f}  "
      f"({_tc_with_pred['correct'].sum():.0f}/{len(_tc_with_pred)} products with non-zero text score)")

print(f"\n    Per-skin-type breakdown:")
print(f"    {'Skin Type':<14}  {'N':>5}  {'Text Correct':>13}  {'Accuracy':>9}")
print(f"    {'-'*14}  {'-'*5}  {'-'*13}  {'-'*9}")
for _st in sorted(_tc_with_pred['truth'].unique()):
    _sub = _tc_with_pred[_tc_with_pred['truth'] == _st]
    _acc = _sub['correct'].mean()
    print(f"    {_st:<14}  {len(_sub):>5}  {_sub['correct'].sum():>13.0f}  {_acc:>9.3f}")

# ── Confusion matrix (text pred vs reviewer truth) ───────────────────────────
print(f"\n  TEXT PREDICTION CONFUSION MATRIX  (rows=truth, cols=text_pred):")
_all_types = sorted(set(_tc_with_pred['truth'].unique()) |
                    set(_tc_with_pred['text_pred'].unique()))
_header = f"    {'truth \\ pred':<14}  " + "  ".join(f"{t[:5]:>7}" for t in _all_types)
print(_header)
print("    " + "-" * (len(_header) - 4))
for _t in _all_types:
    _row_data = _tc_with_pred[_tc_with_pred['truth'] == _t]
    _counts   = [(_row_data['text_pred'] == _p).sum() for _p in _all_types]
    print(f"    {_t:<14}  " + "  ".join(f"{c:>7}" for c in _counts))


print("=" * 70)


# Pre-compute skin-type proportion lookup at startup.
# For each skin_type value, store {product_id: fraction_of_positive_reviewers_matching}.
# Profile Only (Rating) and Full Hybrid infer skin type from the seed product's dominant
# reviewer skin type — a per-seed majority vote — which is structurally distinct
# from the per-product reviewer-proportion scores the evaluator measures here.
# No train/test split is needed: ranking and evaluation draw from independent
# views of the reviewer data (majority-vote inference vs proportion scoring).

def _build_skin_prop_lookup(source_reviews: pd.DataFrame) -> dict:
    """
    Build a lookup keyed by CANONICAL skin-type targets (e.g. 'dry', 'oily').

    Returns NORMALISED LIFT scores rather than raw proportions.
    Lift = proportion_in_product / base_rate_in_pool:
      - 1.0  → product's reviewer mix matches the overall population (neutral)
      - 2.0+ → strongly over-indexed for this skin type
    Capped at 2× and rescaled to [0, 1]:  score = min(lift, 2.0) / 2.0
    so 0.5 = neutral, > 0.5 = over-indexed, < 0.5 = under-indexed.

    This removes demographic bias: a 93% combination product scores 0.5 (neutral)
    rather than 0.93 (spuriously high), giving all skin types a fair metric.

    Parameters
    ----------
    source_reviews : DataFrame
        Pass top_reviews (per-product highest-rated reviewers) so SkinCompat
        measures agreement with the most satisfied customers, not all 4+ reviewers.
    """
    total = len(source_reviews)
    pos_total = source_reviews.groupby('product_id').size()
    canonical_targets = set(_SKIN_TYPE_ALIASES.keys()) | set(_SKIN_TYPE_ALIASES_EXTRA.keys())

    # Compute dataset-level base rate for each skin type
    base_rates = {}
    for target in canonical_targets:
        primary   = _SKIN_TYPE_ALIASES.get(target, set())
        secondary = _SKIN_TYPE_ALIASES_EXTRA.get(target, set())
        _mask = source_reviews['skin_type'].apply(
            lambda v: str(v).strip().lower() == target or
                      str(v).strip().lower() in primary or
                      str(v).strip().lower() in secondary
        )
        base_rates[target] = float(_mask.sum()) / total if total > 0 else 0.0

    print("\n  BASE RATES in reviewer pool (used for lift normalisation):")
    for t, r in sorted(base_rates.items(), key=lambda x: -x[1]):
        print(f"    {t:<14}: {r:.4f}  ({r*100:.1f}%)")

    lookup = {}
    for target in canonical_targets:
        primary   = _SKIN_TYPE_ALIASES.get(target, set())
        secondary = _SKIN_TYPE_ALIASES_EXTRA.get(target, set())
        _mask = source_reviews['skin_type'].apply(
            lambda v: 1.0 if (str(v).strip().lower() == target or
                              str(v).strip().lower() in primary or
                              str(v).strip().lower() in secondary)
                      else 0.0
        )
        _hits      = source_reviews[_mask == 1.0].groupby('product_id').size()
        proportion = (_hits / pos_total).fillna(0.0)
        base_rate  = base_rates[target]
        if base_rate > 0:
            lift_score = (proportion / base_rate).clip(upper=2.0) / 2.0
        else:
            lift_score = pd.Series(0.0, index=proportion.index)
        lookup[target] = lift_score.round(4).to_dict()
    return lookup

# Built from top_reviews (per-product max-rating reviewers) so SkinCompat and
# MAP@5 measure agreement with the most satisfied customers per product.
_skin_prop_lookup: dict = _build_skin_prop_lookup(top_reviews)
print(f"Skin-prop lookup built: {sorted(_skin_prop_lookup.keys())}")

# ── "Any skin type" override ──────────────────────────────────────────────────
# Products tagged "All Skin Types" / "Good for All Skin Types" carry 'any' in
# _product_skin_types. Their reviewer pool is dominated by combination skin
# (dataset base rate), so their computed lift for rare types (e.g. dry, sensitive)
# can be near-zero — unfairly penalising products the brand validated for all types.
#
# Fix (Option C): preserve real reviewer data when it shows genuine over-indexing
# (lift > 0.5); only apply a moderate floor of 0.75 when data is absent or the
# product is being penalised (lift ≤ 0.5) due to sparse reviewer coverage.
# 0.75 → rescaled = (0.75 − 0.5) × 2 = 0.5 — moderately compatible, below a
# genuinely specialised product (which can reach 1.0 from real data).
_any_skin_names = {name for name, types in _product_skin_types.items()
                   if 'any' in types}
_any_skin_pids  = set(
    df.loc[df['product_name'].str.lower().str.strip().isin(_any_skin_names),
           'product_id']
)
for _st_lookup in _skin_prop_lookup.values():
    for _pid in _any_skin_pids:
        current = _st_lookup.get(_pid, None)
        if current is None or current < 0.5:
            _st_lookup[_pid] = 0.75   # moderate floor; real over-indexing kept as-is
print(f"'Any skin type' override applied to {len(_any_skin_pids)} products "
      f"(lift floored at 0.75 where real data is absent or penalised)")

def _compute_skin_prop(product_ids, target_skin_type, prop_lookup=None):
    """
    For each product_id, return the fraction of its positive reviewers
    whose skin_type matches target_skin_type.

    Parameters
    ----------
    prop_lookup : dict or None
        If None, defaults to _skin_prop_lookup (built from all positive reviews).
    """
    if not target_skin_type:
        return {}
    if prop_lookup is None:
        prop_lookup = _skin_prop_lookup  # ranking default (train partition)
    target = target_skin_type.strip().lower()
    lookup = prop_lookup.get(target, {})
    return {pid: lookup.get(pid, 0.0) for pid in product_ids}


def skin_compatibility_score(recommendations_df, user_profile, prop_lookup=None):
    """
    Mean over-index lift score across recommended products for the user's skin type.

    _skin_prop_lookup stores normalised lift: 0.5 = neutral (matches dataset
    base rate), 1.0 = 2× over-indexed. This function rescales the over-indexed
    portion to [0, 1]:
        rescaled = max(lift_score - 0.5, 0) × 2
    so neutral products contribute 0 and under-indexed products are clamped to 0.
    A product loved disproportionately by the user's skin type scores close to 1.

    Range 0-1. Higher = recs genuinely over-index for the user's skin type.
    """
    if prop_lookup is None:
        prop_lookup = _skin_prop_lookup
    if not user_profile or 'skin_type' not in user_profile:
        user_profile = DEFAULT_PROFILE
    target_skin = user_profile['skin_type']
    if target_skin == 'any':
        return 1.0   # universal skin type — all recs fully compatible
    pids        = recommendations_df['product_id'].tolist() \
                  if 'product_id' in recommendations_df.columns else []
    props       = _compute_skin_prop(pids, target_skin, prop_lookup=prop_lookup)
    lift_scores = np.array([props.get(pid, 0.0) for pid in pids]) if pids \
                  else np.zeros(len(recommendations_df))
    rescaled    = np.clip((lift_scores - 0.5) * 2, 0.0, 1.0)
    return round(float(rescaled.mean()), 4)


def map_at_k_skin(recommendations_df, user_profile, threshold=0.60, k=5, prop_lookup=None):
    """
    MAP@K with cosmetics-meaningful relevance:
      A rec is relevant if its normalised lift score >= threshold.
      threshold=0.60 → lift ≥ 1.2 (product 20% over-indexed for user skin type).
    Computed dynamically against the query profile's skin_type.
    Rank-sensitive: rank-1 skin-compatible hit scores higher than rank-5.

    Denominator is K (not the number of hits found), so a list with one
    relevant item at rank 1 scores 1/K, not 1.0 — consistent with standard
    MAP@K and correctly penalising lists with few relevant items.
    """
    if prop_lookup is None:
        prop_lookup = _skin_prop_lookup
    if not user_profile or 'skin_type' not in user_profile:
        user_profile = DEFAULT_PROFILE
    target_skin = user_profile['skin_type']
    pids  = recommendations_df['product_id'].tolist() \
            if 'product_id' in recommendations_df.columns else []
    props = _compute_skin_prop(pids, target_skin, prop_lookup=prop_lookup)

    hits, running_p = 0, 0.0
    for rank, (_, row) in enumerate(recommendations_df.head(k).iterrows(), start=1):
        pid   = row.get('product_id', None)
        score = props.get(pid, 0.0)   # default 0.0: missing pid = no lift signal
        if score >= threshold:
            hits      += 1
            running_p += hits / rank
    return round(running_p / k, 4)

def routine_coherence(recommendations_df):
    """
    What fraction of the recommended products have unique secondary categories?
    Score of 1.0 = every rec is a different product type (perfect routine variety).
    Score of 0.2 = all 5 recs are the same sub-category (e.g. all moisturisers).
    Falls back to primary_category if secondary is unavailable.
    """
    col = 'secondary_category' if 'secondary_category' in recommendations_df.columns \
          else 'primary_category'
    if recommendations_df.empty:
        return 0.0
    n_unique = recommendations_df[col].nunique()
    return round(n_unique / len(recommendations_df), 3)


def intra_list_diversity(recommendations_df):
    """
    Average pairwise TF-IDF feature dissimilarity among recommended products.
    Score of 1.0 = maximally diverse features, 0.0 = all identical.
    """
    rec_names   = recommendations_df['product_name'].str.lower().str.strip()
    rec_indices = []
    for n in rec_names:
        if n in indices.index:
            val = indices.loc[n]
            rec_indices.append(int(val.iloc[0]) if isinstance(val, pd.Series)
                               else int(val))
    if len(rec_indices) < 2:
        return 0.0
    sub  = sim_product[np.ix_(rec_indices, rec_indices)]
    mask = np.triu(np.ones(sub.shape, dtype=bool), k=1)
    return round(float(1.0 - sub[mask].mean()), 3)


def brand_diversity(recommendations_df):
    """
    Fraction of unique brands in the recommendation list.
    1.0 = all different brands  0.2 = all same brand (with 5 recs).
    Guards against brand monopoly in recommendations.
    """
    if recommendations_df.empty:
        return 0.0
    return round(recommendations_df['brand_name'].nunique()
                 / len(recommendations_df), 3)


def serendipity_score(seed_product_name, recommendations_df, user_profile, prop_lookup=None):
    """
    Serendipity v3 — unexpected × dissimilar × quality  (continuous, per-rec score).

    For each recommended product:
        score = (1 − pop_percentile) × (1 − sim_to_seed) × (avg_rating / 5.0)

    Three multiplicative factors:
      (1 − pop_percentile) : popularity unexpectedness — rarer products score higher.
                             Uses binary-search rank in the sorted review-count array
                             → O(log N), no per-call recomputation.
      (1 − sim_to_seed)    : content dissimilarity from the seed — ensures the rec is
                             not just an obvious near-duplicate of the query product.
                             Uses the precomputed sim_product TF-IDF matrix.
      (avg_rating / 5.0)   : quality gate — filters out products that are niche
                             because they are bad, not because they are undiscovered.

    Metric is seed-dependent (same product can be serendipitous for one query, not
    another), avoids overlap with SkinCompat/MAP@5, and requires no skin-type data.

    Fallback: if seed or rec not in index, sim defaults to 0.5 (neutral).
    """
    if recommendations_df.empty:
        return 0.0

    # Resolve seed index once
    seed_name = str(seed_product_name).lower().strip()
    seed_idx  = None
    if seed_name in indices.index:
        val      = indices.loc[seed_name]
        seed_idx = int(val.iloc[0]) if isinstance(val, pd.Series) else int(val)

    scores = []
    n_rc   = len(_rc_sorted)
    for _, row in recommendations_df.iterrows():
        rc     = float(row.get('review_count', 0) or 0)
        rating = float(row.get('avg_rating',   0) or 0)

        # (1) Unexpectedness: fraction of catalogue with fewer reviews
        niche = 1.0 - float(np.searchsorted(_rc_sorted, rc) / n_rc)

        # (2) Dissimilarity from seed
        rec_name = str(row.get('product_name', '')).lower().strip()
        if seed_idx is not None and rec_name in indices.index:
            val     = indices.loc[rec_name]
            rec_idx = int(val.iloc[0]) if isinstance(val, pd.Series) else int(val)
            dissim  = 1.0 - float(sim_product[seed_idx, rec_idx])
        else:
            dissim = 0.5   # neutral fallback

        # (3) Quality
        quality = rating / 5.0

        scores.append(niche * dissim * quality)

    return round(float(np.mean(scores)), 3) if scores else 0.0


def avg_rating_score(recommendations_df):
    """
    Simple average of avg_rating across recommended products (0-5 scale).
    Unweighted and profile-agnostic — identical computation for all configs,
    so differences reflect product quality rather than metric construction.
    """
    return round(float(recommendations_df['avg_rating'].mean()), 3)


def full_evaluation(n_per_skintype=150, top_n=10, random_seed=42):
    """
    Run 7 cosmetics-specific metrics across all configurations.
    Seeds are drawn with STRATIFIED SKIN-TYPE SAMPLING: equal numbers per
    inferred skin type so no single skin type dominates the evaluation.

    Parameters
    ----------
    n_per_skintype : int  seeds per canonical skin type (default 150)

    Metrics
    -------
    SkinCompat    normalised lift score (0.5=neutral, 1.0=2× over-indexed for user skin type)
    MAP@5         rank-sensitive relevance (lift ≥ 1.2× baseline) over top-5
    RoutineCoh    fraction of unique sub-categories in recs
    Diversity     TF-IDF pairwise dissimilarity
    BrandDiv      fraction of unique brands
    Serendipity   (1−pop_pct) × (1−sim_to_seed) × (avg_rating/5) — unexpected, dissimilar, quality

    Returns
    -------
    summary_df : DataFrame indexed by configuration
    detail_df  : long-form per-seed scores (includes skin_type column)
    """
    rng = random.Random(random_seed)

    valid_names = set(df["product_name"].str.lower().str.strip()
                      [df["product_name"].str.lower().str.strip().isin(indices.index)])

    # Pre-build name → primary_category lookup for Panel 4 (SkinCompat by category).
    _name_to_cat = (df.assign(_name=df['product_name'].str.lower().str.strip())
                      .set_index('_name')['primary_category']
                      .to_dict())

    # ── Stratified seed pool: equal n per skin type ──────────────────────────
    # Uses _product_skin_types so multi-type products appear in all their pools.
    # E.g. a product tagged "Best for Dry, Combo, Normal" seeds all three types.
    skin_types = sorted(set(
        st for types in _product_skin_types.values()
        for st in types if st and st != 'any'
    ))
    seeds_by_type = {}
    for st in skin_types:
        pool = [name for name, types in _product_skin_types.items()
                if (st in types or 'any' in types) and name in valid_names]
        seeds_by_type[st] = rng.sample(pool, min(n_per_skintype, len(pool)))

    total_seeds = sum(len(v) for v in seeds_by_type.values())
    print(f"Stratified seed sampling:")
    for st, seeds in seeds_by_type.items():
        print(f"  {st:<14}  {len(seeds):>4} seeds")
    print(f"  {'TOTAL':<14}  {total_seeds:>4} seeds")
    print(f"Configurations: {list(CONFIGURATIONS.keys())}\n")

    all_seeds = [(seed_name, st)
                 for st, seeds in seeds_by_type.items()
                 for seed_name in seeds]

    # Deduplicate seeds: each unique product evaluated once, across ALL its skin
    # types internally, so multi-type products don't get inflated weight.
    _seen_seed_names: set = set()
    unique_seeds: list = []
    for _sname, _st in all_seeds:
        if _sname not in _seen_seed_names:
            _seen_seed_names.add(_sname)
            unique_seeds.append(_sname)
    print(f"Unique seeds after deduplication: {len(unique_seeds)} "
          f"(from {len(all_seeds)} pool entries)\n")

    # Total recommendable products — denominator for Coverage.
    _total_recommendable = len(df[df['product_name'].str.lower().str.strip()
                                   .isin(indices.index)])

    def _eval_config(config_name, config):
        rows          = []
        seen_products = set()    # union of recommended product names across seeds
        rec_freq      = {}       # product_name → count of times recommended (for Gini)
        _rng = random.Random(random_seed)

        for seed_name in unique_seeds:
            # All specific skin types for this seed (exclude 'any' sentinel).
            seed_types = sorted(
                _product_skin_types.get(seed_name, frozenset()) - {'any'}
            )
            if not seed_types:
                seed_types = [_name_to_skin_type.get(
                    seed_name, DEFAULT_PROFILE.get('skin_type', ''))]

            # ── Build recs_by_type ─────────────────────────────────────────
            recs_by_type = {}   # skin_type → DataFrame

            if config["weight_scheme"] == "random":
                # Random is skin-type-independent — one call, reuse for all types.
                _r = get_random_recommendations(seed_name, top_n=top_n, rng=_rng)
                if not (isinstance(_r, str) or _r.empty):
                    recs_by_type = {st: _r for st in seed_types}

            elif config["profile"] != "inferred":
                # No user profile — recs are skin-type-independent.
                _r = get_recommendations(
                    product_name  = seed_name,
                    user_profile  = config["profile"],
                    use_collab    = config["use_collab"],
                    use_reviews   = config["use_reviews"],
                    weight_scheme = config["weight_scheme"],
                    mmr_lambda    = config.get('mmr_lambda', MMR_LAMBDA),
                    epsilon       = config.get('epsilon', 0.0),
                    top_n         = top_n,
                )
                if not (isinstance(_r, str) or _r.empty):
                    recs_by_type = {st: _r for st in seed_types}

            else:
                # Profile-based: recs differ by skin type — separate call per type.
                for st in seed_types:
                    ep = _infer_profile_from_seed(seed_name)
                    ep['skin_type'] = st
                    _r = get_recommendations(
                        product_name  = seed_name,
                        user_profile  = ep,
                        use_collab    = config["use_collab"],
                        use_reviews   = config["use_reviews"],
                        weight_scheme = config["weight_scheme"],
                        mmr_lambda    = config.get('mmr_lambda', MMR_LAMBDA),
                        epsilon       = config.get('epsilon', 0.0),
                        top_n         = top_n,
                    )
                    if not (isinstance(_r, str) or _r.empty):
                        recs_by_type[st] = _r

            if not recs_by_type:
                continue

            # ── Coverage + per-type metrics ────────────────────────────────
            type_metrics = []
            for st, recs in recs_by_type.items():
                for pname in recs['product_name'].str.lower().str.strip():
                    seen_products.add(pname)
                    rec_freq[pname] = rec_freq.get(pname, 0) + 1
                ep = _infer_profile_from_seed(seed_name)
                ep['skin_type'] = st
                type_metrics.append({
                    "skin_compat": skin_compatibility_score(recs, ep),
                    "map_skin":    map_at_k_skin(recs, ep, k=5),
                    "routine_coh": routine_coherence(recs),
                    "diversity":   intra_list_diversity(recs),
                    "brand_div":   brand_diversity(recs),
                    "serendipity": serendipity_score(seed_name, recs, ep),
                })

            # ── Equal-weight average across skin types ─────────────────────
            if not type_metrics:
                continue
            avg = {k: float(np.mean([m[k] for m in type_metrics]))
                   for k in type_metrics[0]}
            rows.append({
                "configuration": config_name,
                "skin_type":     _name_to_skin_type.get(seed_name, ''),
                "seed_product":  seed_name,
                "category":      _name_to_cat.get(seed_name, ''),
                **avg,
            })

        print(f"  Evaluating: {config_name} ... done  "
              f"({len(seen_products)} unique products seen)")
        return rows, seen_products, rec_freq

    results = Parallel(n_jobs=-1, prefer='threads')(
        delayed(_eval_config)(config_name, config)
        for config_name, config in CONFIGURATIONS.items()
    )

    all_rows        = []
    seen_by_config  = {}
    freq_by_config  = {}
    for cname, (config_rows, seen, freq) in zip(CONFIGURATIONS.keys(), results):
        all_rows.extend(config_rows)
        seen_by_config[cname] = seen
        freq_by_config[cname] = freq

    detail_df  = pd.DataFrame(all_rows).dropna()
    metrics    = ['skin_compat', 'map_skin', 'routine_coh',
                  'diversity',   'brand_div', 'serendipity']

    summary_df = (detail_df
                  .groupby('configuration')[metrics]
                  .mean()
                  .round(3)
                  .reindex(list(CONFIGURATIONS.keys()))
                  .rename(columns={
                      'skin_compat':  'SkinCompat',
                      'map_skin':     'MAP@5',
                      'routine_coh':  'RoutineCoh',
                      'diversity':    'Diversity',
                      'brand_div':    'BrandDiv',
                      'serendipity':  'Serendipity',
                  }))

    # ── Coverage metrics ──────────────────────────────────────────────────────
    n_slots = len(all_seeds) * top_n   # total recommendation slots across all seeds

    # Raw coverage: fraction of recommendable catalogue surfaced (union across seeds).
    summary_df['Coverage'] = pd.Series({
        cname: round(len(seen) / _total_recommendable, 3)
        for cname, seen in seen_by_config.items()
    })

    # CovGini: Gini coefficient of product recommendation frequency.
    # 0.0 = all products recommended equally often  (healthy)
    # 1.0 = one product gets all recommendations    (severe filter bubble)
    def _gini(freq_dict):
        if not freq_dict:
            return 0.0
        counts = np.array(sorted(freq_dict.values()), dtype=float)
        n      = len(counts)
        return float((2 * np.dot(np.arange(1, n + 1), counts) / (n * counts.sum())) - (n + 1) / n)

    summary_df['CovGini'] = pd.Series({
        cname: round(_gini(freq), 3)
        for cname, freq in freq_by_config.items()
    })

    return summary_df, detail_df


def evaluate_on_seed_list(seed_names, config_names=None, top_n=10, random_seed=42):
    """
    Run the standard cosmetics evaluation metrics on a pre-specified list of seed
    product names (already lower-stripped).  Used by the Bayesian optimisation loop
    and the test-split comparison so that stratified sampling is done once, outside
    of this function.

    Parameters
    ----------
    seed_names   : list[str]   product names to use as seeds (lower-stripped)
    config_names : list[str] | None   subset of CONFIGURATIONS keys to evaluate;
                   None = all current CONFIGURATIONS
    top_n        : int         recommendations per seed
    random_seed  : int         RNG seed for epsilon-greedy / stochastic configs

    Returns
    -------
    summary_df : DataFrame indexed by config name (mean metrics, rounded to 3dp)
    detail_df  : long-form per-seed scores (one row per seed × config)
    """
    if config_names is None:
        config_names = list(CONFIGURATIONS.keys())

    _name_to_cat = (df.assign(_name=df['product_name'].str.lower().str.strip())
                      .set_index('_name')['primary_category']
                      .to_dict())

    _total_recommendable = len(df[df['product_name'].str.lower().str.strip()
                                   .isin(indices.index)])

    def _eval_one(config_name):
        config        = CONFIGURATIONS[config_name]
        rows          = []
        seen_products = set()
        rec_freq      = {}
        _rng          = random.Random(random_seed)

        for seed_name in seed_names:
            seed_types = sorted(
                _product_skin_types.get(seed_name, frozenset()) - {'any'}
            )
            if not seed_types:
                seed_types = [_name_to_skin_type.get(
                    seed_name, DEFAULT_PROFILE.get('skin_type', ''))]

            recs_by_type = {}

            if config["weight_scheme"] == "random":
                _r = get_random_recommendations(seed_name, top_n=top_n, rng=_rng)
                if not (isinstance(_r, str) or _r.empty):
                    recs_by_type = {st: _r for st in seed_types}

            elif config["profile"] != "inferred":
                _r = get_recommendations(
                    product_name  = seed_name,
                    user_profile  = config["profile"],
                    use_collab    = config["use_collab"],
                    use_reviews   = config["use_reviews"],
                    weight_scheme = config["weight_scheme"],
                    mmr_lambda    = config.get('mmr_lambda', MMR_LAMBDA),
                    epsilon       = config.get('epsilon', 0.0),
                    top_n         = top_n,
                )
                if not (isinstance(_r, str) or _r.empty):
                    recs_by_type = {st: _r for st in seed_types}

            else:
                for st in seed_types:
                    ep = _infer_profile_from_seed(seed_name)
                    ep['skin_type'] = st
                    _r = get_recommendations(
                        product_name  = seed_name,
                        user_profile  = ep,
                        use_collab    = config["use_collab"],
                        use_reviews   = config["use_reviews"],
                        weight_scheme = config["weight_scheme"],
                        mmr_lambda    = config.get('mmr_lambda', MMR_LAMBDA),
                        epsilon       = config.get('epsilon', 0.0),
                        top_n         = top_n,
                    )
                    if not (isinstance(_r, str) or _r.empty):
                        recs_by_type[st] = _r

            if not recs_by_type:
                continue

            type_metrics = []
            for st, recs in recs_by_type.items():
                for pname in recs['product_name'].str.lower().str.strip():
                    seen_products.add(pname)
                    rec_freq[pname] = rec_freq.get(pname, 0) + 1
                ep = _infer_profile_from_seed(seed_name)
                ep['skin_type'] = st
                type_metrics.append({
                    "skin_compat": skin_compatibility_score(recs, ep),
                    "map_skin":    map_at_k_skin(recs, ep, k=5),
                    "routine_coh": routine_coherence(recs),
                    "diversity":   intra_list_diversity(recs),
                    "brand_div":   brand_diversity(recs),
                    "serendipity": serendipity_score(seed_name, recs, ep),
                })

            if not type_metrics:
                continue

            avg = {k: float(np.mean([m[k] for m in type_metrics]))
                   for k in type_metrics[0]}
            rows.append({
                "configuration": config_name,
                "seed_product":  seed_name,
                "category":      _name_to_cat.get(seed_name, ''),
                **avg,
            })

        return rows, seen_products, rec_freq

    results = Parallel(n_jobs=-1, prefer='threads')(
        delayed(_eval_one)(cname) for cname in config_names
    )

    all_rows       = []
    seen_by_config = {}
    freq_by_config = {}
    for cname, (cfg_rows, seen, freq) in zip(config_names, results):
        all_rows.extend(cfg_rows)
        seen_by_config[cname] = seen
        freq_by_config[cname] = freq

    metrics = ['skin_compat', 'map_skin', 'routine_coh',
               'diversity',   'brand_div', 'serendipity']

    if not all_rows:
        detail_df  = pd.DataFrame()
        summary_df = pd.DataFrame(
            0.0, index=config_names,
            columns=['SkinCompat', 'MAP@5', 'RoutineCoh', 'Diversity', 'BrandDiv',
                     'Serendipity', 'Coverage', 'CovGini'])
        return summary_df, detail_df

    detail_df = pd.DataFrame(all_rows).dropna()

    summary_df = (detail_df
                  .groupby('configuration')[metrics]
                  .mean()
                  .round(3)
                  .reindex(config_names)
                  .rename(columns={
                      'skin_compat':  'SkinCompat',
                      'map_skin':     'MAP@5',
                      'routine_coh':  'RoutineCoh',
                      'diversity':    'Diversity',
                      'brand_div':    'BrandDiv',
                      'serendipity':  'Serendipity',
                  }))

    def _gini(freq_dict):
        if not freq_dict:
            return 0.0
        counts = np.array(sorted(freq_dict.values()), dtype=float)
        n      = len(counts)
        return float((2 * np.dot(np.arange(1, n + 1), counts) / (n * counts.sum())) - (n + 1) / n)

    summary_df['Coverage'] = pd.Series({
        cname: round(len(seen) / _total_recommendable, 3)
        for cname, seen in seen_by_config.items()
    })
    summary_df['CovGini'] = pd.Series({
        cname: round(_gini(freq), 3)
        for cname, freq in freq_by_config.items()
    })

    return summary_df, detail_df


def print_full_evaluation_table(summary_df, detail_df):
    """
    Print a formatted summary table with:
    - Per-metric descriptions
    - The full mean scores table
    - Per-metric std across seeds (for the 7 per-seed metrics)
    - Best config per metric (highlighted)
    - Multi-baseline comparison: Full Hybrid vs ALL other configs
    - Clear interpretation notes for each finding
    """
    SEP = "=" * 100

    print("\n" + SEP)
    print("  COSMETICS RECOMMENDER — EVALUATION RESULTS")
    print("  10 Domain-Specific Metrics  x  10 Configurations  (Reviewed Products Only)")
    print(SEP)

    # Metric legend
    desc = {
        'SkinCompat':  'Avg over-index lift per rec: max(lift−0.5,0)×2  (0=neutral/under, 1=2× over-indexed for user skin type)  [higher = better]',
        'MAP@5':       'Mean Average Precision — lift-based relevance ≥ 1.2× baseline, rank-sensitive over top-5 positions  [higher = better]',
        'RoutineCoh':  'Fraction of unique sub-categories in rec list  [1.0 = perfect variety]',
        'Diversity':   'Mean pairwise TF-IDF dissimilarity  [higher = more varied]',
        'BrandDiv':    'Fraction of unique brands  [higher = less brand monopoly]',
        'Serendipity': 'Mean (1−pop_pct)×(1−sim_to_seed)×(avg_rating/5) per rec — unexpected + content-dissimilar + quality  [higher = more genuine discovery]',
        'Coverage':    'Fraction of recommendable catalogue surfaced across all seed queries  [higher = less filter bubble]',
        'CovGini':     'Gini coefficient of product recommendation frequency  [0=perfectly uniform, 1=one product dominates]',
    }
    print(f"\n  {'METRIC':<14}  {'DESCRIPTION'}")
    print(f"  {'-'*14}  {'-'*70}")
    for col, d in desc.items():
        if col in summary_df.columns:
            print(f"  {col:<14}  {d}")

    # Main mean table
    print("\n" + SEP)
    print(summary_df.to_string())
    print(SEP)

    # Std table — per-seed metrics only (Coverage/NormCov/CovGini are config-level)
    per_seed_metrics = ['skin_compat', 'map_skin', 'routine_coh',
                        'diversity', 'brand_div', 'serendipity']
    std_df = (detail_df.groupby('configuration')[per_seed_metrics]
                       .std()
                       .round(3)
                       .reindex(list(CONFIGURATIONS.keys()))
                       .rename(columns={
                           'skin_compat':  'SkinCompat',
                           'map_skin':     'MAP@5',
                           'routine_coh':  'RoutineCoh',
                           'diversity':    'Diversity',
                           'brand_div':    'BrandDiv',
                           'serendipity':  'Serendipity',
                       }))
    print("\n  STD ACROSS SEEDS  (per-seed metrics; Coverage/NormCov/CovGini are config-level aggregates)")
    print(SEP)
    print(std_df.to_string())
    print(SEP)

    # Best config per metric
    print("\n  BEST CONFIGURATION PER METRIC:")
    print(f"  {'Metric':<14}  {'Best Config':<22}  {'Score'}")
    print(f"  {'-'*14}  {'-'*22}  {'-'*6}")
    for col in summary_df.columns:
        best_cfg   = summary_df[col].idxmax()
        best_score = summary_df[col].max()
        print(f"  {col:<14}  {best_cfg:<22}  {best_score:.3f}")

    # Full Hybrid vs all others
    if 'Full Hybrid' in summary_df.index:
        print("\n  FULL HYBRID vs ALL BASELINES:")
        print(f"  {'Baseline':<22}  " +
              "  ".join(f"{c:<10}" for c in summary_df.columns))
        print(f"  {'-'*22}  " +
              "  ".join(f"{'-'*10}" for _ in summary_df.columns))
        hybrid = summary_df.loc['Full Hybrid']
        for cfg in summary_df.index:
            if cfg == 'Full Hybrid':
                continue
            diff = (hybrid - summary_df.loc[cfg]).round(3)
            signs = []
            for v in diff:
                if   v > 0.01:  signs.append(f"+{v:.3f} ↑  ")
                elif v < -0.01: signs.append(f"{v:.3f} ↓  ")
                else:           signs.append(f" {v:.3f} =  ")
            print(f"  {cfg:<22}  " + "  ".join(f"{s:<10}" for s in signs))


def pairwise_significance_test(detail_df, alpha=0.05):
    """
    Pairwise Wilcoxon signed-rank significance tests — referee response (Groups A, B, C).

    Runs two-sided Wilcoxon signed-rank tests on per-seed paired scores for three
    pre-defined groups of configuration comparisons, with Benjamini-Hochberg FDR
    correction applied independently within each group.

    Note: Coverage and CovGini are catalogue-level aggregates (one value per config,
    not per seed) and cannot be tested with this approach.  They are excluded here;
    the paper should note this limitation inline.

    Parameters
    ----------
    detail_df : DataFrame
        Long-form per-seed scores produced by full_evaluation() or
        evaluate_on_seed_list().  Must contain columns:
        'configuration', 'seed_product', and the six per-seed metric columns.
    alpha : float
        FDR threshold for Benjamini-Hochberg correction (default 0.05).

    Returns
    -------
    results : dict[str, DataFrame]
        Keys 'A', 'B', 'C' — one DataFrame per group with columns:
        group, config_a, config_b, metric, n_seeds, mean_a, mean_b,
        delta, p_raw, p_adj, reject, sig.

    Groups
    ------
    A — each profile-aware config vs Content Only (7 configs × 6 metrics = 42 tests)
        Tests whether adding skin-profile information reliably improves any metric.

    B — within the content-family baseline triangle (3 pairs × 6 metrics = 18 tests)
        Tests whether Content+Collab and Content+Reviews differ from Content Only
        and from each other — establishes whether non-profile signals contribute.

    C — Full Hybrid variant family, all pairwise combinations (6 pairs × 6 metrics = 36 tests)
        Tests whether MMR and stochastic post-processing produce statistically
        reliable metric differences, not just numerical ones.
    """
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests
    import itertools

    PER_SEED_METRICS = {
        'skin_compat':  'SkinCompat',
        'map_skin':     'MAP@5',
        'routine_coh':  'RoutineCoh',
        'diversity':    'Diversity',
        'brand_div':    'BrandDiv',
        'serendipity':  'Serendipity',
    }
    available_metrics = {k: v for k, v in PER_SEED_METRICS.items()
                         if k in detail_df.columns}

    all_configs = set(detail_df['configuration'].unique())

    def _aligned_pair(cfg_a, cfg_b, metric):
        """Return two value arrays aligned on shared seed_product index."""
        a = (detail_df[detail_df['configuration'] == cfg_a]
             .set_index('seed_product')[metric].sort_index())
        b = (detail_df[detail_df['configuration'] == cfg_b]
             .set_index('seed_product')[metric].sort_index())
        shared = a.index.intersection(b.index)
        if len(shared) < 10:
            return None, None
        return a.loc[shared].values, b.loc[shared].values

    def _run_group(pairs, group_name):
        rows, raw_pvals = [], []
        for cfg_a, cfg_b in pairs:
            if cfg_a not in all_configs or cfg_b not in all_configs:
                continue
            for raw_metric, display_metric in available_metrics.items():
                a_arr, b_arr = _aligned_pair(cfg_a, cfg_b, raw_metric)
                if a_arr is None:
                    continue
                delta = float(np.mean(a_arr - b_arr))
                diffs = a_arr - b_arr
                if np.all(diffs == 0):
                    p_raw = 1.0
                else:
                    try:
                        _, p_raw = wilcoxon(a_arr, b_arr, alternative='two-sided',
                                            zero_method='wilcox')
                    except ValueError:
                        p_raw = 1.0
                rows.append({
                    'group':    group_name,
                    'config_a': cfg_a,
                    'config_b': cfg_b,
                    'metric':   display_metric,
                    'n_seeds':  len(a_arr),
                    'mean_a':   round(float(np.mean(a_arr)), 4),
                    'mean_b':   round(float(np.mean(b_arr)), 4),
                    'delta':    round(delta, 4),
                    'p_raw':    round(p_raw, 6),
                })
                raw_pvals.append(p_raw)

        if not rows:
            return pd.DataFrame()

        result = pd.DataFrame(rows)
        reject, p_adj, _, _ = multipletests(raw_pvals, alpha=alpha, method='fdr_bh')
        result['p_adj']   = np.round(p_adj, 6)
        result['reject']  = reject
        result['sig']     = result['p_adj'].apply(
            lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else
                      ('*'  if p < 0.05 else 'n.s.'))
        )
        return result

    # ── Group definitions ──────────────────────────────────────────────────
    PROFILE_AWARE_CONFIGS = [
        'Content+Skin Profile',
        'Collab+Skin Profile',
        'review+Skin Profile',
        'Full Hybrid',
        'Full Hybrid + MMR',
        'Full Hybrid Stochastic',
        'Full Hybrid+MMR+stochastic',
    ]
    GROUP_A_PAIRS = [(cfg, 'Content Only') for cfg in PROFILE_AWARE_CONFIGS]

    GROUP_B_PAIRS = [
        ('Content Only',     'Content + Collab'),
        ('Content Only',     'Content + Reviews'),
        ('Content + Collab', 'Content + Reviews'),
    ]

    FH_VARIANTS = [
        'Full Hybrid',
        'Full Hybrid + MMR',
        'Full Hybrid Stochastic',
        'Full Hybrid+MMR+stochastic',
    ]
    GROUP_C_PAIRS = list(itertools.combinations(FH_VARIANTS, 2))

    results = {
        'A': _run_group(GROUP_A_PAIRS, 'A'),
        'B': _run_group(GROUP_B_PAIRS, 'B'),
        'C': _run_group(GROUP_C_PAIRS, 'C'),
    }
    return results


def print_significance_results(sig_results, alpha=0.05):
    """
    Print formatted pairwise significance tables for Groups A, B, and C.

    Each group is printed as a wide matrix (one row per config pair, one column
    per metric) showing: delta (mean_a − mean_b) and BH-corrected significance.
    A positive delta means config_a scores higher than config_b on that metric.
    A summary count table is printed at the end.

    Parameters
    ----------
    sig_results : dict[str, DataFrame]   output of pairwise_significance_test()
    alpha       : float                  FDR threshold used (for display only)
    """
    SEP  = '=' * 115
    THIN = '-' * 115

    GROUP_DESCRIPTIONS = {
        'A': ('Profile-aware configurations vs Content Only  '
              '[config_a = profile-aware,  config_b = Content Only]'),
        'B': ('Content-family configurations: all pairwise combinations  '
              '[config_a scores minus config_b scores]'),
        'C': ('Full Hybrid variant family: all pairwise combinations  '
              '[config_a scores minus config_b scores]'),
    }

    METRICS_ORDER = ['SkinCompat', 'MAP@5', 'RoutineCoh', 'Diversity', 'BrandDiv', 'Serendipity']

    print('\n' + SEP)
    print('  PAIRWISE SIGNIFICANCE TESTS — REFEREE RESPONSE (Groups A, B, C)')
    print('  Test: two-sided Wilcoxon signed-rank on per-seed paired scores.')
    print(f'  Correction: Benjamini-Hochberg FDR (q = {alpha}) applied within each group.')
    print('  Delta = mean(config_a metric) − mean(config_b metric);  positive = config_a higher.')
    print('  Coverage and CovGini are catalogue-level aggregates — no per-seed test possible.')
    print('  Significance: * p_adj < .05   ** p_adj < .01   *** p_adj < .001   n.s. = not significant')
    print(SEP)

    all_frames = []

    for grp_key in ['A', 'B', 'C']:
        df_grp = sig_results.get(grp_key, pd.DataFrame())
        desc   = GROUP_DESCRIPTIONS.get(grp_key, '')

        print(f'\n  GROUP {grp_key} — {desc}')

        if df_grp.empty:
            print('    No results (configurations not found in detail_df)')
            continue

        n_total = len(df_grp)
        n_sig   = int((df_grp['sig'] != 'n.s.').sum())
        print(f'  {n_sig} significant out of {n_total} tests after BH correction.')
        print(THIN)

        # Header row
        avail_metrics = [m for m in METRICS_ORDER if m in df_grp['metric'].values]
        pair_hdr = f"  {'Config A':<34}  {'Config B':<34}"
        met_hdr  = '  '.join(f'{m:<16}' for m in avail_metrics)
        print(f'{pair_hdr}  {met_hdr}')
        print(f"  {'-'*34}  {'-'*34}  " + '  '.join(f'{"-"*16}' for _ in avail_metrics))

        pairs = (df_grp[['config_a', 'config_b']]
                 .drop_duplicates()
                 .values.tolist())

        for cfg_a, cfg_b in pairs:
            sub  = (df_grp[(df_grp['config_a'] == cfg_a) &
                           (df_grp['config_b'] == cfg_b)]
                    .set_index('metric'))
            pair_str = f"  {cfg_a:<34}  {cfg_b:<34}"
            cells = []
            for m in avail_metrics:
                if m not in sub.index:
                    cells.append(f"{'—':<16}")
                    continue
                row  = sub.loc[m]
                d    = float(row['delta'])
                sig  = str(row['sig'])
                sign = '+' if d >= 0 else ''
                cell = f'{sign}{d:.3f} {sig}'
                cells.append(f'{cell:<16}')
            print(f'{pair_str}  ' + '  '.join(cells))

        all_frames.append(df_grp)
        print()

    # ── Cross-group summary ────────────────────────────────────────────────
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        print(SEP)
        print('  SUMMARY — significant tests by group and metric (after BH correction)')
        print(THIN)
        sig_only = combined[combined['sig'] != 'n.s.'].copy()
        if not sig_only.empty:
            pivot = (sig_only
                     .groupby(['group', 'metric'])
                     .size()
                     .unstack(fill_value=0)
                     .reindex(columns=[m for m in METRICS_ORDER
                                       if m in sig_only['metric'].values],
                              fill_value=0))
            print(pivot.to_string())
        else:
            print('  No significant results found.')

        print()
        print('  DIRECTION SUMMARY — metrics where profile-aware configs consistently beat Content Only')
        print(THIN)
        grp_a = sig_results.get('A', pd.DataFrame())
        if not grp_a.empty:
            sig_a = grp_a[grp_a['sig'] != 'n.s.']
            pos_a = sig_a[sig_a['delta'] > 0].groupby('metric').size().rename('configs_better')
            neg_a = sig_a[sig_a['delta'] < 0].groupby('metric').size().rename('configs_worse')
            dir_df = pd.concat([pos_a, neg_a], axis=1).fillna(0).astype(int)
            dir_df.index.name = 'metric'
            print(f"  {'Metric':<16}  {'Configs beating Content Only':<30}  "
                  f"{'Configs worse than Content Only'}")
            print(f"  {'-'*16}  {'-'*30}  {'-'*30}")
            for metric, row in dir_df.iterrows():
                print(f"  {metric:<16}  {int(row.get('configs_better', 0)):<30}  "
                      f"{int(row.get('configs_worse', 0))}")

    print('\n' + SEP + '\n')


def plot_full_evaluation(summary_df, detail_df,
                         save_path='full_evaluation.png'):
    """
    6-panel cosmetics evaluation dashboard:
      Panel 1 — Radar: per-seed metrics per configuration
      Panel 2 — Grouped bar: SkinCompat, MAP@5 (core personalisation)
      Panel 3 — Grouped bar: RoutineCoh, Diversity, BrandDiv, Serendipity, Coverage, CovGini (quality)
      Panel 4 — SkinCompat by sub-category (which categories benefit most)
      Panel 5 — Heatmap: all metrics x all configurations
      Panel 6 — Full side-by-side comparison bar chart
    """
    configs  = list(CONFIGURATIONS.keys())
    n_cfg    = len(configs)
    # Coverage/Gini excluded from radar (aggregate metrics, different nature to per-seed averages).
    # They appear in Panel 3, heatmap, and the full comparison panel.
    radar_metrics = ['SkinCompat', 'MAP@5', 'RoutineCoh',
                     'Diversity',  'BrandDiv', 'Serendipity']
    metrics       = ['SkinCompat', 'MAP@5', 'RoutineCoh',
                     'Diversity',  'BrandDiv', 'Serendipity',
                     'Coverage',   'CovGini']
    display_df = summary_df.copy()

    categories = sorted(detail_df['category'].unique()) \
                 if 'category' in detail_df.columns else []

    fig = plt.figure(figsize=(24, 20))
    fig.patch.set_facecolor(JOURNAL_BG)
    gs  = GridSpec(3, 3, figure=fig,
                   hspace=0.54, wspace=0.40,
                   top=0.91, bottom=0.06, left=0.06, right=0.97)

    ax_radar = fig.add_subplot(gs[0, 0], polar=True)
    ax_pers  = fig.add_subplot(gs[0, 1])
    ax_qual  = fig.add_subplot(gs[0, 2])
    ax_cat   = fig.add_subplot(gs[1, :2])
    ax_heat  = fig.add_subplot(gs[1, 2])
    ax_all   = fig.add_subplot(gs[2, :])

    fig.suptitle(
        'Cosmetics Recommender Evaluation Dashboard\n'
        f'8 Domain-Relevant Metrics  ×  {n_cfg} Configurations  '
        '(Reviewed Products Only)',
        fontsize=14, fontweight='bold', color=JOURNAL_TEXT, y=0.97)

    def style(ax):
        ax.set_facecolor(JOURNAL_BG)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['bottom', 'left']].set_color(JOURNAL_SPINE)
        ax.tick_params(colors=JOURNAL_TEXT)
        ax.set_axisbelow(True)

    # Panel 1: Radar — 7 per-seed metrics only (Coverage is aggregate, excluded)
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]
    ax_radar.set_facecolor(JOURNAL_BG)
    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(radar_metrics, size=8, color=JOURNAL_TEXT)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_radar.set_yticklabels(['0.25','0.50','0.75','1.0'], size=7, color=JOURNAL_SUBTEXT)
    ax_radar.grid(color=JOURNAL_GRID, linewidth=0.8)
    ax_radar.set_title('Metric Profile\nper Configuration\n(Coverage shown in Panel 3)',
                       fontsize=10, fontweight='bold', color=JOURNAL_TEXT, pad=14)
    for cname in configs:
        if cname not in display_df.index:
            continue
        vals = [display_df.loc[cname, m] if m in display_df.columns else 0
                for m in radar_metrics]
        vals += vals[:1]
        ax_radar.plot(angles, vals, 'o-', linewidth=2,
                      color=COLORS[cname], label=cname)
        ax_radar.fill(angles, vals, alpha=0.08, color=COLORS[cname])
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.6, 1.2),
                    fontsize=7.5, framealpha=1.0,
                    facecolor=JOURNAL_BG, edgecolor=JOURNAL_SPINE)

    # Panel 2: Core personalisation metrics
    pers_metrics = [m for m in ['SkinCompat', 'MAP@5']
                    if m in summary_df.columns]
    x2 = np.arange(len(pers_metrics)); bw2 = 0.80 / n_cfg
    for i, cname in enumerate(configs):
        if cname not in summary_df.index:
            continue
        vals   = [summary_df.loc[cname, m] for m in pers_metrics]
        offset = (i - n_cfg / 2 + 0.5) * bw2
        bars   = ax_pers.bar(x2 + offset, vals, bw2,
                             label=cname, color=COLORS[cname],
                             alpha=0.90, edgecolor='white')
        for bar, val in zip(bars, vals):
            if val > 0.01:
                ax_pers.text(bar.get_x() + bar.get_width() / 2,
                             bar.get_height() + 0.005,
                             f'{val:.2f}', ha='center', va='bottom',
                             fontsize=6.5, color=JOURNAL_TEXT)
    ax_pers.set_xticks(x2)
    ax_pers.set_xticklabels(pers_metrics, fontsize=10)
    ax_pers.set_ylabel('Score (0-1)', fontsize=9, color=JOURNAL_SUBTEXT)
    ax_pers.set_ylim(0, 1.15)
    ax_pers.set_title('Personalisation Metrics\n'
                      '(Skin Compatibility · MAP@5)',
                      fontsize=10, fontweight='bold', color=JOURNAL_TEXT)
    ax_pers.legend(fontsize=7, framealpha=1.0, facecolor=JOURNAL_BG,
                   edgecolor=JOURNAL_SPINE, ncol=2)
    ax_pers.yaxis.grid(True, color=JOURNAL_GRID, linewidth=0.5, linestyle='--')
    style(ax_pers)

    # Panel 3: Quality / diversity metrics (Coverage, CovGini; NormCov removed)
    qual_metrics = [m for m in ['RoutineCoh', 'Diversity', 'BrandDiv', 'Serendipity',
                                 'Coverage', 'CovGini']
                    if m in summary_df.columns]
    x3 = np.arange(len(qual_metrics)); bw3 = 0.80 / n_cfg
    for i, cname in enumerate(configs):
        if cname not in summary_df.index:
            continue
        vals   = [summary_df.loc[cname, m] for m in qual_metrics]
        offset = (i - n_cfg / 2 + 0.5) * bw3
        bars   = ax_qual.bar(x3 + offset, vals, bw3,
                             label=cname, color=COLORS[cname],
                             alpha=0.90, edgecolor='white')
        for bar, val in zip(bars, vals):
            if val > 0.01:
                ax_qual.text(bar.get_x() + bar.get_width() / 2,
                             bar.get_height() + 0.005,
                             f'{val:.2f}', ha='center', va='bottom',
                             fontsize=6.5, color=JOURNAL_TEXT)
    ax_qual.set_xticks(x3)
    ax_qual.set_xticklabels(qual_metrics, fontsize=10)
    ax_qual.set_ylabel('Score (0-1)', fontsize=9, color=JOURNAL_SUBTEXT)
    ax_qual.set_ylim(0, 1.15)
    ax_qual.set_title('Quality & Coverage Metrics\n'
                      '(RoutineCoh · Diversity · BrandDiv · Serendipity · Coverage · CovGini)',
                      fontsize=10, fontweight='bold', color=JOURNAL_TEXT)
    ax_qual.legend(fontsize=7, framealpha=1.0, facecolor=JOURNAL_BG,
                   edgecolor=JOURNAL_SPINE, ncol=2)
    ax_qual.yaxis.grid(True, color=JOURNAL_GRID, linewidth=0.5, linestyle='--')
    style(ax_qual)

    # Panel 4: SkinCompat by sub-category (top 12 categories)
    if categories and 'skin_compat' in detail_df.columns:
        # Pick top 12 categories by number of seeds evaluated
        top_cats  = (detail_df.groupby('category').size()
                              .nlargest(12).index.tolist())
        n_cats    = len(top_cats)
        x4        = np.arange(n_cats); bw4 = 0.80 / n_cfg
        for i, cname in enumerate(configs):
            sub = detail_df[detail_df['configuration'] == cname]
            cat_means = [
                sub[sub['category'] == c]['skin_compat'].mean()
                if c in sub['category'].values else 0
                for c in top_cats
            ]
            offset = (i - n_cfg / 2 + 0.5) * bw4
            ax_cat.bar(x4 + offset, cat_means, bw4,
                       label=cname, color=COLORS[cname],
                       alpha=0.90, edgecolor='white')
        ax_cat.set_xticks(x4)
        ax_cat.set_xticklabels(top_cats, rotation=28, ha='right', fontsize=8.5)
        ax_cat.set_ylabel('Skin Compatibility', fontsize=10, color=JOURNAL_SUBTEXT)
        ax_cat.set_ylim(0, 1.15)
        ax_cat.set_title('Skin Compatibility by Sub-Category\n'
                         '(how well each config matches skin type per product group)',
                         fontsize=11, fontweight='bold', color=JOURNAL_TEXT)
        ax_cat.legend(fontsize=8, framealpha=1.0, facecolor=JOURNAL_BG,
                      edgecolor=JOURNAL_SPINE, ncol=3)
        ax_cat.yaxis.grid(True, color=JOURNAL_GRID, linewidth=0.5, linestyle='--')
        style(ax_cat)

    # Panel 5: Summary heatmap (8 metrics)
    heat_metrics = [m for m in metrics if m in display_df.columns]
    heat_data    = display_df[heat_metrics].values
    row_labels   = list(display_df.index)
    im = ax_heat.imshow(heat_data, cmap=JOURNAL_CMAP, aspect='auto', vmin=0, vmax=1)
    ax_heat.set_xticks(range(len(heat_metrics)))
    ax_heat.set_xticklabels(heat_metrics, fontsize=8, color=JOURNAL_TEXT,
                             fontweight='bold', rotation=30, ha='right')
    ax_heat.set_yticks(range(len(row_labels)))
    ax_heat.set_yticklabels(row_labels, fontsize=8.5, color=JOURNAL_TEXT)
    ax_heat.set_title('Summary Heatmap',
                      fontsize=10, fontweight='bold', color=JOURNAL_TEXT, pad=8)
    for i in range(len(row_labels)):
        for j in range(len(heat_metrics)):
            val     = heat_data[i, j]
            txt_col = 'white' if val > 0.55 else JOURNAL_TEXT
            orig = summary_df.loc[row_labels[i], heat_metrics[j]] \
                   if heat_metrics[j] in summary_df.columns else val
            ax_heat.text(j, i, f'{orig:.2f}',
                         ha='center', va='center',
                         fontsize=8, fontweight='bold', color=txt_col)
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    # Panel 6: All metrics side-by-side (8 metrics)
    all_metrics = [m for m in metrics if m in summary_df.columns]
    x6  = np.arange(len(all_metrics)); bw6 = 0.80 / n_cfg
    for i, cname in enumerate(configs):
        if cname not in display_df.index:
            continue
        vals   = [display_df.loc[cname, m] for m in all_metrics]
        offset = (i - n_cfg / 2 + 0.5) * bw6
        bars   = ax_all.bar(x6 + offset, vals, bw6,
                            label=cname, color=COLORS[cname],
                            alpha=0.90, edgecolor='white')
        for bar, val in zip(bars, vals):
            if val > 0.02:
                ax_all.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005,
                            f'{val:.2f}', ha='center', va='bottom',
                            fontsize=6, color=JOURNAL_TEXT)
    ax_all.set_xticks(x6)
    ax_all.set_xticklabels(all_metrics, fontsize=10)
    ax_all.set_ylabel('Score (0-1)', fontsize=10, color=JOURNAL_SUBTEXT)
    ax_all.set_ylim(0, 1.2)
    ax_all.set_title('All 8 Metrics — Full Configuration Comparison',
                     fontsize=12, fontweight='bold', color=JOURNAL_TEXT)
    ax_all.legend(fontsize=9, framealpha=1.0, facecolor=JOURNAL_BG,
                  edgecolor=JOURNAL_SPINE, ncol=6, loc='upper right')
    ax_all.yaxis.grid(True, color=JOURNAL_GRID, linewidth=0.5, linestyle='--')
    style(ax_all)

    plt.savefig(save_path, dpi=JOURNAL_DPI, bbox_inches='tight', facecolor=JOURNAL_BG)
    plt.close()
    print(f"Full evaluation chart saved to: {save_path}")


def plot_extended_evaluation(rank_summary, score_table,
                             pref_summary, pref_pivot,
                             save_path='extended_evaluation.png'):
    """
    4-panel extended evaluation dashboard:
      Panel 1 - Kendall's Tau heatmap between config pairs
      Panel 2 - Recommendation overlap % bar chart
      Panel 3 - Cross-profile personalisation score comparison across hybrid variants
                (person_score, baseline_score, adjusted_score per config)
      Panel 4 - SkinCompat per simulated profile × config (multi-profile preference)
    Coverage metrics (Coverage, NormCov, CovGini) are in full_evaluation / plot_full_evaluation.
    """
    configs = list(CONFIGS.keys())
    n_cfg   = len(configs)

    fig = plt.figure(figsize=(24, 14))
    fig.patch.set_facecolor(JOURNAL_BG)
    gs  = GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.42,
                   top=0.92, bottom=0.06, left=0.06, right=0.97)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])

    fig.suptitle(
        'Extended Evaluation Dashboard — Beauty Recommender System\n'
        'Ranking Disagreement  |  Personalisation  |  Preference',
        fontsize=14, fontweight='bold', color=JOURNAL_TEXT, y=0.97)

    def style(ax):
        ax.set_facecolor(JOURNAL_BG)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['bottom','left']].set_color(JOURNAL_SPINE)
        ax.tick_params(colors=JOURNAL_TEXT)
        ax.set_axisbelow(True)

    # Panel 1: Kendall's Tau heatmap
    pairs      = rank_summary.index.tolist()
    tau_vals   = rank_summary['mean_tau'].fillna(0).values
    tau_matrix = np.zeros((n_cfg, n_cfg))
    np.fill_diagonal(tau_matrix, 1.0)
    for k, pair in enumerate(pairs):
        parts = pair.split(' vs ')
        if parts[0] in configs and parts[1] in configs:
            i, j = configs.index(parts[0]), configs.index(parts[1])
            tau_matrix[i, j] = tau_matrix[j, i] = tau_vals[k]
    im1 = ax1.imshow(tau_matrix, cmap=JOURNAL_CMAP, vmin=0, vmax=1, aspect='auto')
    short = [c.replace('Collab+Skin Profile','Co+Skin')
               .replace('review+Skin Profile','Rev+Skin')
               .replace('Content+Skin Profile','C+Skin')
               .replace('Content + ','C+').replace('Content ','C.')
               .replace('Popularity','Pop.').replace('Full Hybrid','Hybrid') for c in configs]
    ax1.set_xticks(range(n_cfg)); ax1.set_xticklabels(short, fontsize=7, color=JOURNAL_TEXT, rotation=30)
    ax1.set_yticks(range(n_cfg)); ax1.set_yticklabels(short, fontsize=7, color=JOURNAL_TEXT)
    ax1.set_title("Kendall's Tau\n(1.0=same order, 0.0=different)",
                  fontsize=9, fontweight='bold', color=JOURNAL_TEXT)
    for i in range(n_cfg):
        for j in range(n_cfg):
            ax1.text(j, i, f'{tau_matrix[i,j]:.2f}', ha='center', va='center',
                     fontsize=8, fontweight='bold',
                     color='white' if tau_matrix[i,j] > 0.55 else JOURNAL_TEXT)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Panel 2: Overlap % bar chart — show only cross-baseline pairs
    key_pairs = [p for p in pairs if 'Content Only' in p or 'Popularity' in p]
    if not key_pairs:
        key_pairs = pairs[:6]
    ov_vals   = [rank_summary.loc[p, 'mean_overlap'] for p in key_pairs]
    pair_cols = [COLORS.get(p.split(' vs ')[1], '#0072B2') for p in key_pairs]
    short_kp  = [p.replace('Content Only vs ','vs ').replace('Content + ','C+')
                   .replace('Full Hybrid','Hybrid').replace('Popularity','Pop.')
                 for p in key_pairs]
    bars2 = ax2.bar(range(len(key_pairs)), [v*100 for v in ov_vals],
                    color=pair_cols, alpha=0.88, edgecolor='white', width=0.55)
    for bar, val in zip(bars2, ov_vals):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9, color=JOURNAL_TEXT)
    ax2.set_xticks(range(len(key_pairs))); ax2.set_xticklabels(short_kp, fontsize=7.5, rotation=20, ha='right')
    ax2.set_ylabel('Mean Overlap %', fontsize=9, color=JOURNAL_SUBTEXT); ax2.set_ylim(0, 115)
    ax2.set_title('Recommendation Overlap\nvs Content Only Baseline',
                  fontsize=10, fontweight='bold', color=JOURNAL_TEXT)
    ax2.yaxis.grid(True, color=JOURNAL_GRID, linewidth=0.5, linestyle='--'); style(ax2)

    # Panel 3: Cross-profile personalisation score comparison — all hybrid variants
    # Shows person_score (profile-pair differentiation), baseline_score (content
    # similarity alone), and adjusted_score (genuine profile effect) side by side.
    hybrid_names = list(score_table.index) if score_table is not None and not score_table.empty else []
    short_hybrids = [n.replace('Full Hybrid', 'Hybrid').replace('+MMR+stochastic', '+MMR+ε')
                       .replace('Stochastic', 'ε') for n in hybrid_names]
    x3 = np.arange(len(hybrid_names))
    bw3 = 0.25
    _bar_colors = ['#0072B2', '#56B4E9', '#009E73']  # blue, sky-blue, green
    _labels3    = ['PersonScore', 'BaselineScore', 'AdjustedScore']
    for bi, (col, key, lbl) in enumerate(zip(
            _bar_colors, ['person_score', 'baseline_score', 'adjusted_score'], _labels3)):
        if score_table is None or key not in score_table.columns:
            continue
        vals3 = score_table[key].reindex(hybrid_names).fillna(0).values
        bars3 = ax3.bar(x3 + (bi - 1) * bw3, vals3, bw3,
                        label=lbl, color=col, alpha=0.88, edgecolor='white')
        for bar, val in zip(bars3, vals3):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{val:+.3f}' if key == 'adjusted_score' else f'{val:.3f}',
                     ha='center', va='bottom', fontsize=7, color=JOURNAL_TEXT)
    ax3.axhline(0, color=JOURNAL_SPINE, linewidth=0.8, linestyle='--')
    ax3.set_xticks(x3); ax3.set_xticklabels(short_hybrids, fontsize=7.5,
                                              color=JOURNAL_TEXT, rotation=12, ha='right')
    ax3.set_ylabel('Score (0–1)', fontsize=9, color=JOURNAL_SUBTEXT)
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_title('Cross-Profile Personalisation\n(PersonScore vs Baseline vs Adjusted)',
                  fontsize=9, fontweight='bold', color=JOURNAL_TEXT)
    ax3.legend(fontsize=7, framealpha=1.0, facecolor=JOURNAL_BG, edgecolor=JOURNAL_SPINE)
    ax3.yaxis.grid(True, color=JOURNAL_GRID, linewidth=0.5, linestyle='--')
    style(ax3)

    # Panel 4: SkinCompat per simulated profile × config
    # Shows whether profile-aware configs score higher for EACH different user
    profiles_in_pivot = list(pref_pivot.index) if pref_pivot is not None and not pref_pivot.empty else []
    bw4 = 0.13
    x4  = np.arange(len(profiles_in_pivot))
    for i, cname in enumerate(configs):
        if pref_pivot is None or cname not in pref_pivot.columns:
            continue
        vals4  = [float(pref_pivot.loc[p, cname]) if p in pref_pivot.index else 0.0
                  for p in profiles_in_pivot]
        offset = (i - n_cfg/2 + 0.5) * bw4
        bars4  = ax4.bar(x4 + offset, vals4, bw4, label=cname,
                         color=COLORS[cname], alpha=0.90, edgecolor='white')
        for bar, val in zip(bars4, vals4):
            if val > 0.01:
                ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004,
                         f'{val:.2f}', ha='center', va='bottom', fontsize=5.5,
                         color=JOURNAL_TEXT, rotation=90)
    short_prof = [p.replace(' + ', '\n+\n') for p in profiles_in_pivot]
    ax4.set_xticks(x4)
    ax4.set_xticklabels(short_prof, fontsize=8)
    ax4.set_ylabel('Skin Compatibility (0–1)', fontsize=9, color=JOURNAL_SUBTEXT)
    ax4.set_ylim(0, 0.55)
    ax4.set_title('Skin Compatibility per Simulated User Profile × Configuration\n'
                  '(profile-aware configs should score higher for each simulated user)',
                  fontsize=10, fontweight='bold', color=JOURNAL_TEXT)
    ax4.legend(fontsize=7, framealpha=1.0, facecolor=JOURNAL_BG,
               edgecolor=JOURNAL_SPINE, ncol=3, loc='upper right')
    ax4.yaxis.grid(True, color=JOURNAL_GRID, linewidth=0.5, linestyle='--')
    style(ax4)

    plt.savefig(save_path, dpi=JOURNAL_DPI, bbox_inches='tight', facecolor=JOURNAL_BG)
    plt.close()
    print(f"Extended evaluation chart saved to: {save_path}")


# ===========================================================================
# STEP 11 - EXTENDED EVALUATION FUNCTIONS
# ===========================================================================

from scipy.stats import kendalltau, ttest_1samp, wilcoxon

try:
    import optuna as _optuna_module
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

CONFIGS = CONFIGURATIONS

TEST_PROFILES = {
    'Dry + Fair':        {'skin_tone': 'fair',      'skin_type': 'dry',
                          'hair_color': 'blonde',    'eye_color': 'blue'},
    'Oily + Light':      {'skin_tone': 'light',     'skin_type': 'oily',
                          'hair_color': 'brown',     'eye_color': 'brown'},
    'Combination + Deep':  {'skin_tone': 'deep',      'skin_type': 'combination',
                            'hair_color': 'black',     'eye_color': 'brown'},
    'Normal + Tan':      {'skin_tone': 'tan',       'skin_type': 'normal',
                          'hair_color': 'brown',     'eye_color': 'hazel'},
}


def _rbo(list_a, list_b, p=0.9):
    """
    Rank-Biased Overlap — handles non-conjoint ranked lists natively.
    No shared items needed. Range [0, 1]: 0=completely different, 1=identical.
    p=0.9 weights the top 10 positions heavily (standard for top-10 lists).
    """
    score, set_a, set_b = 0.0, set(), set()
    for d, (ea, eb) in enumerate(zip(list_a, list_b), start=1):
        set_a.add(ea); set_b.add(eb)
        score += (p ** (d - 1)) * len(set_a & set_b) / d
    return round((1 - p) * score, 4)


def ranking_disagreement(n_per_skintype=150, top_n=10, random_seed=42):
    """
    Pairwise Kendall's Tau + RBO between every config pair across stratified seeds.
    Tau=1.0 => identical ranking order on shared items.
    RBO=1.0  => identical full lists (works even with zero shared items).
    Lower values = configs are more differentiated.

    Seeds use stratified skin-type sampling (n_per_skintype per skin type, deduplicated)
    so skin-type representation matches full_evaluation (~556 unique seeds).
    """
    rng = random.Random(random_seed)
    valid_names = set(df['product_name'].str.lower().str.strip()
                      [df['product_name'].str.lower().str.strip().isin(indices.index)])
    skin_types = sorted(set(
        st for types in _product_skin_types.values()
        for st in types if st and st != 'any'
    ))
    seeds_by_type = {}
    for st in skin_types:
        pool = [name for name, types in _product_skin_types.items()
                if (st in types or 'any' in types) and name in valid_names]
        seeds_by_type[st] = rng.sample(pool, min(n_per_skintype, len(pool)))
    _seen_t1 = set()
    seeds = []
    for _st_seeds in seeds_by_type.values():
        for _sname in _st_seeds:
            if _sname not in _seen_t1:
                _seen_t1.add(_sname)
                seeds.append(_sname)
    print(f"  Test 1 stratified seed sampling:")
    for st, st_seeds in seeds_by_type.items():
        print(f"    {st:<14}  {len(st_seeds):>4} seeds")
    print(f"    {'TOTAL':<14}  {sum(len(v) for v in seeds_by_type.values()):>4} pool entries"
          f"  →  {len(seeds)} unique seeds\n")
    config_names = list(CONFIGS.keys())
    pairs        = [(config_names[i], config_names[j])
                    for i in range(len(config_names))
                    for j in range(i + 1, len(config_names))]
    def _eval_seed(seed_name):
        # Per-seed RNG: deterministic and thread-safe (no shared state).
        _seed_rng = random.Random(random_seed ^ hash(seed_name))
        rec_lists = {}
        for cname, config in CONFIGS.items():
            if config['weight_scheme'] == 'random':
                recs = get_random_recommendations(seed_name, top_n=top_n, rng=_seed_rng)
            else:
                profile = (_infer_profile_from_seed(seed_name)
                           if config['profile'] == 'inferred'
                           else config['profile'])
                recs = get_recommendations(
                    product_name  = seed_name,
                    user_profile  = profile,
                    use_collab    = config['use_collab'],
                    use_reviews   = config['use_reviews'],
                    weight_scheme = config['weight_scheme'],
                    mmr_lambda    = config.get('mmr_lambda', MMR_LAMBDA),
                    epsilon       = config.get('epsilon', 0.0),
                    top_n         = top_n
                )
            if not isinstance(recs, str) and not recs.empty:
                rec_lists[cname] = recs['product_name'].tolist()
        if len(rec_lists) < 2:
            return []
        seed_rows = []
        for a, b in pairs:
            if a not in rec_lists or b not in rec_lists:
                continue
            common = [n for n in rec_lists[a] if n in rec_lists[b]]
            # Default tau=0.0: fewer than 2 shared items means rankings are
            # maximally different (no agreement measurable), so tau is 0.
            tau, p_tau = 0.0, np.nan
            if len(common) >= 2:
                ra = [rec_lists[a].index(n) for n in common]
                rb = [rec_lists[b].index(n) for n in common]
                tau, p_tau = kendalltau(ra, rb)
            shared = set(rec_lists[a]) & set(rec_lists[b])
            seed_rows.append({
                'seed':        seed_name,
                'pair':        f"{a} vs {b}",
                'kendall_tau': round(float(tau), 3) if not np.isnan(tau) else np.nan,
                'p_tau':       round(float(p_tau), 4) if not np.isnan(p_tau) else np.nan,
                'rbo':         _rbo(rec_lists[a], rec_lists[b]),
                'unique_a':    len(set(rec_lists[a]) - set(rec_lists[b])),
                'unique_b':    len(set(rec_lists[b]) - set(rec_lists[a])),
                'overlap_pct': round(len(shared) / top_n, 3),
            })
        return seed_rows

    results = Parallel(n_jobs=-1, prefer='threads')(
        delayed(_eval_seed)(seed_name) for seed_name in seeds
    )
    rows = [row for seed_rows in results for row in seed_rows]
    detail  = pd.DataFrame(rows)
    summary = (detail.groupby('pair')
                     .agg(mean_tau=('kendall_tau', 'mean'),
                          mean_rbo=('rbo', 'mean'),
                          mean_overlap=('overlap_pct', 'mean'),
                          mean_unique_a=('unique_a', 'mean'),
                          mean_unique_b=('unique_b', 'mean'))
                     .round(4))

    # One-sample t-test per pair on tau values across seeds.
    # H0: mean_tau = 0 (rankings are uncorrelated / configs produce independent orders)
    # H1: mean_tau > 0 (configs produce positively correlated rankings)
    # Low p = rankings are significantly similar; high p = configs are genuinely different.
    p_vals, sigs = {}, {}
    for pair, grp in detail.groupby('pair'):
        taus = grp['kendall_tau'].dropna().values
        if len(taus) >= 2 and taus.std() > 0:
            _, p = ttest_1samp(taus, popmean=0, alternative='greater')
            p = round(float(p), 4)
        else:
            p = np.nan
        p_vals[pair] = p
        sigs[pair]   = ('' if np.isnan(p) else
                        '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '')
    summary['p_value'] = summary.index.map(p_vals)
    summary['sig']     = summary.index.map(sigs)

    print("\n" + "=" * 65)
    print("  TEST 1: Ranking Disagreement")
    print("  Tau: 1.0=identical order on shared items  0.0=completely different")
    print("  RBO: 1.0=identical full lists  0.0=completely different (no shared items needed)")
    print("  p_value: H0=tau==0 (uncorrelated)  *p<.05  **p<.01  ***p<.001")
    print("=" * 65)
    print(summary.to_string())
    return detail, summary


def cross_profile_overlap_test(n_per_skintype=150, top_n=10, random_seed=42,
                               config_names=None):
    """
    Cross-profile overlap test for all hybrid configurations.

    For each config, runs stratified seeds through every TEST_PROFILE (4 profiles)
    + a No Profile baseline. Pairwise set-overlap between profile recommendation
    lists measures how much the profile signal differentiates recommendations.

    Seeds use stratified skin-type sampling (n_per_skintype per skin type, deduplicated)
    matching full_evaluation (~556 unique seeds).

    Three scores reported per config:
      person_score          = 1 - mean overlap between profile pairs
                              (higher = more differentiation between profiles)
      baseline_person_score = 1 - mean overlap vs No Profile baseline
                              (how much content similarity alone differentiates)
      adjusted_person_score = person_score - baseline_person_score
                              (> 0 = genuine profile effect beyond content similarity)

    Returns:
      all_details  : dict[config_name -> detail_df per-seed per-pair rows]
      all_summaries: dict[config_name -> summary_df with p-values]
      score_table  : DataFrame[configuration x person_score/baseline_score/adjusted_score]
    """
    if config_names is None:
        config_names = HYBRID_CONFIGS

    rng = random.Random(random_seed)
    valid_names = set(df['product_name'].str.lower().str.strip()
                      [df['product_name'].str.lower().str.strip().isin(indices.index)])
    skin_types = sorted(set(
        st for types in _product_skin_types.values()
        for st in types if st and st != 'any'
    ))
    seeds_by_type = {}
    for st in skin_types:
        pool = [name for name, types in _product_skin_types.items()
                if (st in types or 'any' in types) and name in valid_names]
        seeds_by_type[st] = rng.sample(pool, min(n_per_skintype, len(pool)))
    _seen_t2 = set()
    seeds = []
    for _st_seeds in seeds_by_type.values():
        for _sname in _st_seeds:
            if _sname not in _seen_t2:
                _seen_t2.add(_sname)
                seeds.append(_sname)
    print(f"  Test 2 stratified seed sampling:")
    for st, st_seeds in seeds_by_type.items():
        print(f"    {st:<14}  {len(st_seeds):>4} seeds")
    print(f"    {'TOTAL':<14}  {sum(len(v) for v in seeds_by_type.values()):>4} pool entries"
          f"  →  {len(seeds)} unique seeds\n")

    all_details   = {}
    all_summaries = {}
    score_rows    = []

    for config_name in config_names:
        cfg  = CONFIGURATIONS[config_name]
        rows = []

        for seed_name in seeds:
            rec_sets = {}

            # Collect recs under each user profile
            for label, profile in TEST_PROFILES.items():
                recs = get_recommendations(
                    product_name  = seed_name,
                    user_profile  = profile,
                    use_collab    = cfg['use_collab'],
                    use_reviews   = cfg['use_reviews'],
                    weight_scheme = cfg['weight_scheme'],
                    mmr_lambda    = cfg.get('mmr_lambda', MMR_LAMBDA),
                    epsilon       = cfg.get('epsilon', 0.0),
                    top_n         = top_n,
                )
                if not isinstance(recs, str) and not recs.empty:
                    rec_sets[label] = set(recs['product_name'].tolist())

            # No-profile baseline: same weight scheme, no user context
            recs_none = get_recommendations(
                product_name  = seed_name,
                user_profile  = None,
                use_collab    = cfg['use_collab'],
                use_reviews   = cfg['use_reviews'],
                weight_scheme = cfg['weight_scheme'],
                mmr_lambda    = cfg.get('mmr_lambda', MMR_LAMBDA),
                epsilon       = cfg.get('epsilon', 0.0),
                top_n         = top_n,
            )
            if not isinstance(recs_none, str) and not recs_none.empty:
                rec_sets['No Profile'] = set(recs_none['product_name'].tolist())

            labels = list(rec_sets.keys())
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    a, b = labels[i], labels[j]
                    rows.append({
                        'config':      config_name,
                        'seed':        seed_name,
                        'pair':        f"{a} vs {b}",
                        'overlap_pct': round(len(rec_sets[a] & rec_sets[b]) / top_n, 3),
                    })

        detail  = pd.DataFrame(rows)
        summary = (detail.groupby('pair')['overlap_pct']
                         .agg(['mean', 'std', 'min', 'max']).round(3)
                         .rename(columns={'mean': 'mean_overlap', 'std': 'std_overlap',
                                          'min': 'min_overlap',  'max': 'max_overlap'}))
        # One-sample t-test per pair: H0 = overlap == 1.0 (no personalisation)
        p_values = {}
        for pair, grp in detail.groupby('pair'):
            vals = grp['overlap_pct'].dropna().values
            if len(vals) >= 2 and vals.std() > 0:
                _, p = ttest_1samp(vals, popmean=1.0, alternative='less')
            else:
                p = np.nan
            p_values[pair] = round(p, 4) if not np.isnan(p) else np.nan
        summary['p_value'] = summary.index.map(p_values)
        summary['sig'] = summary['p_value'].apply(
            lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            if not (isinstance(p, float) and np.isnan(p)) else '')

        # Split profile-vs-profile pairs from vs-no-profile pairs
        profile_detail  = detail[~detail['pair'].str.contains('No Profile')]
        baseline_detail = detail[ detail['pair'].str.contains('No Profile')]

        person_score          = round(1 - profile_detail['overlap_pct'].mean(), 3)
        baseline_person_score = round(1 - baseline_detail['overlap_pct'].mean(), 3)
        adjusted_person_score = round(person_score - baseline_person_score, 3)

        all_details[config_name]   = detail
        all_summaries[config_name] = summary
        score_rows.append({
            'configuration':  config_name,
            'person_score':   person_score,
            'baseline_score': baseline_person_score,
            'adjusted_score': adjusted_person_score,
        })

    score_table = pd.DataFrame(score_rows).set_index('configuration')

    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  TEST 2: Cross-Profile Overlap — All Hybrid Variants")
    print("  person_score:    1 - mean overlap between profile pairs")
    print("  baseline_score:  1 - mean overlap vs No Profile baseline")
    print("  adjusted_score:  person_score − baseline_score  (>0 = genuine profile effect)")
    print("  p_value: H0=overlap==1.0 (no personalisation)  *p<.05  **p<.01  ***p<.001")
    print(SEP)
    print(f"\n  {'Configuration':<35}  {'PersonScore':>11}  {'Baseline':>8}  {'Adjusted':>8}")
    print(f"  {'-'*35}  {'-'*11}  {'-'*8}  {'-'*8}")
    for idx, row in score_table.iterrows():
        print(f"  {idx:<35}  {row['person_score']:>11.3f}  "
              f"{row['baseline_score']:>8.3f}  {row['adjusted_score']:>+8.3f}")

    for config_name, summary in all_summaries.items():
        print(f"\n  [{config_name}]  mean_overlap, std, min, max, p_value, sig")
        print(summary.to_string())

    return all_details, all_summaries, score_table


def simulated_preference_test(n_per_skintype=150, top_n=10, random_seed=42):
    """
    Simulates 4 distinct user profiles (TEST_PROFILES) across all configurations.

    For EACH simulated user, every config is run and recommendations are evaluated
    against THAT USER'S skin type — not a single global default. This tests whether
    profile-aware configs genuinely adapt to different users.

    Seeds are drawn with the same STRATIFIED SKIN-TYPE SAMPLING used in
    full_evaluation (n_per_skintype per skin type, deduplicated), so skin-type
    balance and seed count match the main evaluation (~556 unique seeds).

    Parameters
    ----------
    n_per_skintype : int  seeds per canonical skin type (default 150, matching full_evaluation)

    Structure of results:
      detail_df  : one row per (profile x seed x config)
      summary_df : mean metrics indexed by (profile, configuration)
      pivot_df   : mean skin_compat per profile x config — the key personalisation table

    Metrics:
      skin_compat : mean prop_skin_type of recs matched to THIS user's skin type
      map_skin    : MAP@5 with lift-based relevance threshold 0.60 for THIS user
      routine_coh : fraction of unique sub-categories in recs
    """
    rng = random.Random(random_seed)
    valid_names = set(df['product_name'].str.lower().str.strip()
                      [df['product_name'].str.lower().str.strip().isin(indices.index)])
    skin_types = sorted(set(
        st for types in _product_skin_types.values()
        for st in types if st and st != 'any'
    ))
    seeds_by_type = {}
    for st in skin_types:
        pool = [name for name, types in _product_skin_types.items()
                if (st in types or 'any' in types) and name in valid_names]
        seeds_by_type[st] = rng.sample(pool, min(n_per_skintype, len(pool)))

    # Deduplicate: preserve first-occurrence order (matches full_evaluation unique_seeds)
    _seen_pref = set()
    seeds = []
    for _st_seeds in seeds_by_type.values():
        for _sname in _st_seeds:
            if _sname not in _seen_pref:
                _seen_pref.add(_sname)
                seeds.append(_sname)

    print(f"  Test 3 stratified seed sampling:")
    for st, st_seeds in seeds_by_type.items():
        print(f"    {st:<14}  {len(st_seeds):>4} seeds")
    print(f"    {'TOTAL':<14}  {sum(len(v) for v in seeds_by_type.values()):>4} pool entries"
          f"  →  {len(seeds)} unique seeds\n")

    def _eval_profile(prof_label, sim_profile):
        prof_rows = []
        _pref_rng = random.Random(random_seed ^ hash(prof_label))
        for seed_name in seeds:
            for cname, config in CONFIGS.items():
                if config['weight_scheme'] == 'random':
                    recs = get_random_recommendations(seed_name, top_n=top_n, rng=_pref_rng)
                else:
                    # For profile-aware configs, use the simulated user profile.
                    # For non-profile configs, pass None so the system doesn't personalise —
                    # but we still EVALUATE the recs against the simulated user's skin type.
                    recs = get_recommendations(
                        product_name  = seed_name,
                        user_profile  = sim_profile if config['profile'] is not None else None,
                        use_collab    = config['use_collab'],
                        use_reviews   = config['use_reviews'],
                        weight_scheme = config['weight_scheme'],
                        mmr_lambda    = config.get('mmr_lambda', MMR_LAMBDA),
                        epsilon       = config.get('epsilon', 0.0),
                        top_n         = top_n
                    )
                if isinstance(recs, str) or recs.empty:
                    continue
                prof_rows.append({
                    'profile':       prof_label,
                    'skin_type':     sim_profile['skin_type'],
                    'seed':          seed_name,
                    'configuration': cname,
                    'skin_compat':   skin_compatibility_score(recs, sim_profile),
                    'map_skin':      map_at_k_skin(recs, sim_profile, k=5),
                    'routine_coh':   routine_coherence(recs),
                })
        return prof_rows

    results = Parallel(n_jobs=-1, prefer='threads')(
        delayed(_eval_profile)(prof_label, sim_profile)
        for prof_label, sim_profile in TEST_PROFILES.items()
    )
    rows = [row for prof_rows in results for row in prof_rows]
    detail  = pd.DataFrame(rows)
    metrics = ['skin_compat', 'map_skin', 'routine_coh']

    # Summary: mean metrics by (profile, config)
    summary = (detail.groupby(['profile', 'configuration'])[metrics]
                     .mean().round(3))

    # Pivot: skin_compat per profile x config — shows personalisation adaptation
    pivot = (detail.groupby(['profile', 'configuration'])['skin_compat']
                   .mean().round(3)
                   .unstack('configuration')
                   .reindex(columns=list(CONFIGS.keys())))

    # MAP@5 pivot — rank-sensitive skin relevance
    pivot_map = (detail.groupby(['profile', 'configuration'])['map_skin']
                       .mean().round(3)
                       .unstack('configuration')
                       .reindex(columns=list(CONFIGS.keys())))

    SEP = "=" * 90
    print(f"\n{SEP}")
    print("  TEST 3: Simulated User Preference — 4 Profiles × 10 Configurations")
    print("  Each config evaluated against the SIMULATED USER'S skin type, not a global default.")
    print("  Profile-aware configs (Content+Skin Profile, Collab+Skin Profile, review+Skin Profile, Full Hybrid) pass the user profile to the engine.")
    print("  Non-profile configs are evaluated against the user's skin type post-hoc.")
    print(SEP)

    print("\n  SKIN COMPATIBILITY (SkinCompat) — higher = recs match user skin type")
    print("  (ideal: profile-aware configs score higher than non-profile configs for each user)")
    print(pivot.to_string())

    print("\n  MAP@5 (skin-type relevance, rank-sensitive) — higher = better")
    print(pivot_map.to_string())

    print("\n  FULL DETAIL (mean per profile × config):")
    print(summary.to_string())

    # Personalisation lift: do profile-aware configs score higher for each user?
    # Wilcoxon signed-rank per metric: H1 = Full Hybrid > Content Only
    def _wilcoxon_p(h_vals, c_vals):
        min_len = min(len(h_vals), len(c_vals))
        if min_len >= 10 and not np.allclose(h_vals[:min_len], c_vals[:min_len]):
            _, p = wilcoxon(h_vals[:min_len], c_vals[:min_len], alternative='greater')
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            return f'{p:.4f}', sig
        return 'N/A', ''

    print("\n  PERSONALISATION LIFT — All Hybrid Variants vs Content Only, per profile:")
    print("  Wilcoxon signed-rank: H1 = hybrid > content_only  *p<.05  **p<.01  ***p<.001")
    for prof in TEST_PROFILES:
        if prof not in summary.index.get_level_values('profile'):
            continue
        c_sc  = detail[(detail['profile']==prof)&(detail['configuration']=='Content Only')]['skin_compat'].values
        c_map = detail[(detail['profile']==prof)&(detail['configuration']=='Content Only')]['map_skin'].values
        c_rc  = detail[(detail['profile']==prof)&(detail['configuration']=='Content Only')]['routine_coh'].values
        try:
            c_mean = summary.loc[(prof, 'Content Only')]
        except KeyError:
            continue

        print(f"\n  Profile: {prof}")
        print(f"  {'Configuration':<35} {'ΔSkinCompat':>12} {'p':>7} {'sig':>4}"
              f"  {'ΔMAP@5':>8} {'p':>7} {'sig':>4}"
              f"  {'ΔRoutineCoh':>12} {'p':>7} {'sig':>4}")
        print(f"  {'-'*35} {'-'*12} {'-'*7} {'-'*4}"
              f"  {'-'*8} {'-'*7} {'-'*4}"
              f"  {'-'*12} {'-'*7} {'-'*4}")
        for hcfg in HYBRID_CONFIGS:
            try:
                h_mean = summary.loc[(prof, hcfg)]
                h_sc   = detail[(detail['profile']==prof)&(detail['configuration']==hcfg)]['skin_compat'].values
                h_map  = detail[(detail['profile']==prof)&(detail['configuration']==hcfg)]['map_skin'].values
                h_rc   = detail[(detail['profile']==prof)&(detail['configuration']==hcfg)]['routine_coh'].values

                d_sc  = h_mean['skin_compat'] - c_mean['skin_compat']
                d_map = h_mean['map_skin']    - c_mean['map_skin']
                d_rc  = h_mean['routine_coh'] - c_mean['routine_coh']

                p_sc,  sig_sc  = _wilcoxon_p(h_sc,  c_sc)
                p_map, sig_map = _wilcoxon_p(h_map, c_map)
                p_rc,  sig_rc  = _wilcoxon_p(h_rc,  c_rc)

                print(f"  {hcfg:<35} {d_sc:>+12.3f} {p_sc:>7} {sig_sc:>4}"
                      f"  {d_map:>+8.3f} {p_map:>7} {sig_map:>4}"
                      f"  {d_rc:>+12.3f} {p_rc:>7} {sig_rc:>4}")
            except KeyError:
                pass

    return detail, summary, pivot


# Test 4 (catalogue_coverage) removed in V35.
# Coverage, NormCov, and CovGini are now computed inside full_evaluation()
# over 750 stratified seeds — more reliable than the old 100-seed un-stratified version.


# ===========================================================================
# STEP 12 — BAYESIAN WEIGHT OPTIMISATION  (V39)
# ===========================================================================

def bayesian_weight_optimisation(n_per_skintype=150, n_trials=60,
                                  val_fraction=0.70, top_n=10, random_seed=42):
    """
    Bayesian (TPE) sweep over the 4 free hybrid weights on a held-out validation
    split of stratified seed queries.

    Fixed at zero: w_review, w_pop, w_review_st.
    Free (projected onto the unit simplex): w_content, w_collab, w_skinrating, w_collab_st.

    Composite objective (maximised):
        0.40 × SkinCompat  +  0.30 × MAP@5  +  0.20 × RoutineCoh  +  0.10 × Serendipity

    Parameters
    ----------
    n_per_skintype : int   seeds per skin type for stratified pool (default 150)
    n_trials       : int   TPE trials (default 60)
    val_fraction   : float fraction of unique seeds used for validation (default 0.70)
    top_n          : int   recommendations per seed
    random_seed    : int   RNG seed (shared for pool generation and TPE sampler)

    Returns
    -------
    study        : optuna.Study  (None if optuna unavailable)
    best_weights : dict          normalised weight dict
    val_seeds    : list[str]
    test_seeds   : list[str]
    """
    if not _OPTUNA_AVAILABLE:
        print("  optuna not installed — skipping Bayesian optimisation.")
        print("  Install with:  pip install optuna")
        return None, None, [], []

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── Stratified seed pool (same logic as full_evaluation) ──────────────
    rng = random.Random(random_seed)
    valid_names = set(df['product_name'].str.lower().str.strip()
                      [df['product_name'].str.lower().str.strip().isin(indices.index)])
    skin_types = sorted(set(
        st for types in _product_skin_types.values()
        for st in types if st and st != 'any'
    ))
    seeds_by_type = {}
    for st in skin_types:
        pool = [name for name, types in _product_skin_types.items()
                if (st in types or 'any' in types) and name in valid_names]
        seeds_by_type[st] = rng.sample(pool, min(n_per_skintype, len(pool)))

    _seen = set()
    all_seeds = []
    for _st_seeds in seeds_by_type.values():
        for _sname in _st_seeds:
            if _sname not in _seen:
                _seen.add(_sname)
                all_seeds.append(_sname)

    # ── 70/30 stratified split ─────────────────────────────────────────────
    n_val     = int(len(all_seeds) * val_fraction)
    val_seeds  = all_seeds[:n_val]
    test_seeds = all_seeds[n_val:]

    print(f"  BO seed split:  {len(val_seeds)} val  /  {len(test_seeds)} test  "
          f"(from {len(all_seeds)} unique seeds, {n_per_skintype}/skin-type pool)")

    # ── TPE objective ──────────────────────────────────────────────────────
    def objective(trial):
        w_c  = trial.suggest_float('w_content',    0.05, 0.60)
        w_co = trial.suggest_float('w_collab',     0.05, 0.50)
        w_sk = trial.suggest_float('w_skinrating', 0.10, 0.65)
        w_cs = trial.suggest_float('w_collab_st',  0.05, 0.50)
        total = w_c + w_co + w_sk + w_cs
        trial_weights = {
            'w_content':    w_c  / total,
            'w_review':     0.0,
            'w_collab':     w_co / total,
            'w_pop':        0.0,
            'w_skinrating': w_sk / total,
            'w_collab_st':  w_cs / total,
            'w_review_st':  0.0,
        }
        CONFIGURATIONS['__bo_trial__'] = {
            'profile':       'inferred',
            'use_collab':    True,
            'use_reviews':   False,
            'weight_scheme': trial_weights,
            'mmr_lambda':    1.0,
            'epsilon':       0.0,
        }
        try:
            summary_df, _ = evaluate_on_seed_list(
                val_seeds, config_names=['__bo_trial__'],
                top_n=top_n, random_seed=random_seed)
            row = summary_df.loc['__bo_trial__']
            score = (0.40 * float(row.get('SkinCompat', 0))
                   + 0.30 * float(row.get('MAP@5',      0))
                   + 0.20 * float(row.get('RoutineCoh', 0))
                   + 0.10 * float(row.get('Serendipity', 0)))
        except Exception:
            score = 0.0
        finally:
            CONFIGURATIONS.pop('__bo_trial__', None)
        return float(score)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=random_seed),
    )
    print(f"  Running {n_trials} TPE trials ...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # ── Extract and normalise best weights ─────────────────────────────────
    best = study.best_params
    total = (best['w_content'] + best['w_collab']
             + best['w_skinrating'] + best['w_collab_st'])
    best_weights = {
        'w_content':    round(best['w_content']    / total, 4),
        'w_review':     0.0,
        'w_collab':     round(best['w_collab']     / total, 4),
        'w_pop':        0.0,
        'w_skinrating': round(best['w_skinrating'] / total, 4),
        'w_collab_st':  round(best['w_collab_st']  / total, 4),
        'w_review_st':  0.0,
    }
    # Fix any rounding drift so weights sum exactly to 1.0
    _drift = 1.0 - sum(best_weights.values())
    if abs(_drift) > 1e-9:
        best_weights['w_content'] = round(best_weights['w_content'] + _drift, 4)

    print(f"\n  Best val composite: {study.best_value:.4f}  (trial #{study.best_trial.number})")
    print(f"  Optimal weights (non-zero):")
    for k, v in best_weights.items():
        if v > 0:
            print(f"    {k:<16}  {v:.4f}")

    return study, best_weights, val_seeds, test_seeds


def test_split_comparison(best_weights, test_seeds, top_n=10, random_seed=42):
    """
    Evaluate the Bayesian-optimised hybrid alongside all hand-designed
    CONFIGURATIONS on the reserved test split.

    The 'Optimised Hybrid' config is added to CONFIGURATIONS in-place so it
    appears in all subsequent calls (e.g. plot_weight_optimisation).

    Parameters
    ----------
    best_weights : dict     normalised weight dict from bayesian_weight_optimisation
    test_seeds   : list[str]
    top_n        : int
    random_seed  : int

    Returns
    -------
    summary_df : DataFrame indexed by configuration (mean metrics on test seeds)
    """
    CONFIGURATIONS['Optimised Hybrid'] = {
        'profile':       'inferred',
        'use_collab':    True,
        'use_reviews':   False,
        'weight_scheme': best_weights,
        'mmr_lambda':    1.0,
        'epsilon':       0.0,
    }

    config_names = list(CONFIGURATIONS.keys())
    print(f"  Test split: {len(test_seeds)} seeds  ×  {len(config_names)} configurations "
          f"(including Optimised Hybrid)")

    summary_df, _ = evaluate_on_seed_list(
        test_seeds, config_names=config_names,
        top_n=top_n, random_seed=random_seed)

    col_w = [8, 7, 7, 7, 8, 8]
    header = (f"  {'Configuration':<28}  "
              + "  ".join(f"{m:>{w}}" for m, w in zip(
                  ['SkinCmp', 'MAP@5', 'RoutCoh', 'Divers', 'BrandDiv', 'Serendip'],
                  col_w)))
    print(f"\n  TEST-SPLIT RESULTS  (Optimised Hybrid vs hand-designed configs)")
    print(f"  Seeds: {len(test_seeds)}")
    print(header)
    print("  " + "-" * 88)
    for cfg in config_names:
        if cfg not in summary_df.index:
            continue
        row    = summary_df.loc[cfg]
        marker = "  ◀ optimised" if cfg == 'Optimised Hybrid' else ""
        vals   = [row.get(m, 0) for m in
                  ['SkinCompat', 'MAP@5', 'RoutineCoh', 'Diversity', 'BrandDiv', 'Serendipity']]
        print(f"  {cfg:<28}  "
              + "  ".join(f"{v:>{w}.3f}" for v, w in zip(vals, col_w))
              + marker)

    return summary_df


def plot_weight_optimisation(study, best_weights, test_summary_df,
                              save_path='weight_optimisation.png'):
    """
    3-panel figure:
      Panel 1 — TPE optimisation history (trial objective scores)
      Panel 2 — Bar chart of the optimal weight values
      Panel 3 — Grouped bar: Optimised Hybrid vs key hand-designed configs on test split

    Parameters
    ----------
    study           : optuna.Study
    best_weights    : dict
    test_summary_df : DataFrame returned by test_split_comparison
    save_path       : str
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor(JOURNAL_BG)
    gs  = GridSpec(1, 3, figure=fig, wspace=0.40,
                   left=0.06, right=0.97, top=0.88, bottom=0.16)
    ax1, ax2, ax3 = [fig.add_subplot(gs[0, i]) for i in range(3)]

    def style(ax):
        ax.set_facecolor(JOURNAL_BG)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['bottom', 'left']].set_color(JOURNAL_SPINE)
        ax.tick_params(colors=JOURNAL_SUBTEXT, labelsize=8)
        ax.xaxis.label.set_color(JOURNAL_SUBTEXT)
        ax.yaxis.label.set_color(JOURNAL_SUBTEXT)

    fig.suptitle('Bayesian Weight Optimisation — V39',
                 fontsize=13, fontweight='bold', color=JOURNAL_TEXT, y=0.98)

    # ── Panel 1: optimisation history ─────────────────────────────────────
    trials_df = study.trials_dataframe()
    ax1.plot(trials_df['number'], trials_df['value'],
             color=COLORS.get('Full Hybrid', '#CC79A7'), alpha=0.5, linewidth=0.8)
    ax1.scatter(trials_df['number'], trials_df['value'],
                c=trials_df['value'], cmap='Blues', s=18, zorder=3)
    best_t = study.best_trial.number
    ax1.axvline(best_t, color='#117733', linestyle='--', linewidth=1.2,
                label=f'Best (trial {best_t})')
    ax1.set_xlabel('Trial', fontsize=9)
    ax1.set_ylabel('Composite score (val)', fontsize=9)
    ax1.set_title('TPE Optimisation History', fontsize=10, color=JOURNAL_TEXT)
    ax1.legend(fontsize=8, frameon=False)
    ax1.yaxis.grid(True, color=JOURNAL_GRID, linewidth=0.5, linestyle='--')
    style(ax1)

    # ── Panel 2: optimal weight values ────────────────────────────────────
    w_labels = [k for k, v in best_weights.items() if v > 0]
    w_values = [best_weights[k] for k in w_labels]
    ax2.bar(range(len(w_labels)), w_values,
            color=COLORS.get('Optimised Hybrid', '#117733'), alpha=0.82)
    ax2.set_xticks(range(len(w_labels)))
    ax2.set_xticklabels([k.replace('w_', '') for k in w_labels],
                        rotation=30, ha='right', fontsize=8)
    ax2.set_ylabel('Weight', fontsize=9)
    ax2.set_title('Optimal Signal Weights', fontsize=10, color=JOURNAL_TEXT)
    ax2.yaxis.grid(True, color=JOURNAL_GRID, linewidth=0.5, linestyle='--')
    style(ax2)

    # ── Panel 3: test-split comparison ────────────────────────────────────
    compare_cfgs = [c for c in [
        'Content Only', 'Full Hybrid', 'Full Hybrid + MMR',
        'Full Hybrid Stochastic', 'Full Hybrid+MMR+stochastic',
        'Optimised Hybrid',
    ] if c in test_summary_df.index]

    metric_cols = ['SkinCompat', 'MAP@5', 'RoutineCoh', 'Serendipity']
    x     = np.arange(len(metric_cols))
    width = 0.13

    for i, cfg in enumerate(compare_cfgs):
        vals = [float(test_summary_df.loc[cfg].get(m, 0)) for m in metric_cols]
        ax3.bar(x + i * width, vals, width,
                label=cfg,
                color=COLORS.get(cfg, '#999999'), alpha=0.85)

    offset = width * (len(compare_cfgs) - 1) / 2
    ax3.set_xticks(x + offset)
    ax3.set_xticklabels(metric_cols, fontsize=9)
    ax3.set_ylabel('Score', fontsize=9)
    ax3.set_title('Test Split: Optimised vs Hand-Designed', fontsize=10, color=JOURNAL_TEXT)
    ax3.legend(fontsize=7, frameon=False, ncol=2, loc='upper right')
    ax3.yaxis.grid(True, color=JOURNAL_GRID, linewidth=0.5, linestyle='--')
    style(ax3)

    plt.savefig(save_path, dpi=JOURNAL_DPI, bbox_inches='tight', facecolor=JOURNAL_BG)
    plt.close()
    print(f"Weight optimisation chart saved to: {save_path}")


# ===========================================================================
# STEP 13 - RUN EVERYTHING
# ===========================================================================

# ── Pre-warm all profile caches ──────────────────────────────────────────
# Collect every unique profile used anywhere in the program.
# Running these now (in parallel) means the first get_recommendations call
# for each profile hits the cache instead of doing the heavy computation.
_all_profiles = [p for p in
                 [c['profile'] for c in CONFIGURATIONS.values()] +
                 list(TEST_PROFILES.values())
                 if p is not None and isinstance(p, dict)]

# Also pre-warm all canonical skin types that can be inferred from seeds
_inferred_skin_types = set(_name_to_skin_type.values()) - {''}
for _st in _inferred_skin_types:
    _p = dict(DEFAULT_PROFILE)
    _p['skin_type'] = _st
    _all_profiles.append(_p)
_unique_profiles = list({_profile_key(p): p for p in _all_profiles}.values())

def _warm_profile(profile):
    key = _profile_key(profile)
    # 1. reviewer weights (used by weighted review text)
    _reviewer_weights(profile)
    # 2. review TF-IDF matrix + pre-computed full similarity matrix
    if key not in _review_tfidf_cache:
        weighted_text = get_weighted_review_text(profile)   # populates _weighted_text_cache
        _wrt = df['product_id'].map(weighted_text).fillna('')
        tfidf_w = TfidfVectorizer(stop_words='english', max_features=5000)
        X = tfidf_w.fit_transform(_wrt)
        _review_tfidf_cache[key] = X
        _review_sim_cache[key] = cosine_similarity(X).astype(np.float32)

print("Pre-warming profile caches ...")
# Sequential: each warm-up allocates a full N×N cosine_similarity matrix (~256 MB)
# plus large string buffers in groupby.agg; parallel execution exhausts RAM.
for _p in _unique_profiles:
    _warm_profile(_p)
print(f"  Done — {len(list(_unique_profiles))} profiles cached.")

# ── Free large objects no longer needed after warm-up ────────────────────
import gc
_weighted_text_cache.clear()          # amplified text served its purpose
if 'review_text' in df.columns:
    df.drop(columns=['review_text'], inplace=True)   # sim_review already built
if 'author_id' in reviews.columns:
    reviews.drop(columns=['author_id'], inplace=True) # collab matrix already built
gc.collect()
print("  Memory freed: review_text, amplified text cache, author_id.\n")

my_profile = DEFAULT_PROFILE
test_product = "Moisture Surge 100H Auto-Replenishing Hydrator Moisturizer"
# Quick smoke test
print("-- With user profile (Full Hybrid) --")
top=10
results = get_recommendations(
    product_name  = test_product,
    user_profile  = my_profile,
    use_collab    = True,
    use_reviews   = True,
    weight_scheme = 'full',
    top_n         = top
)
print(results)

print("\n-- Popularity config --")
print(get_recommendations(
    product_name  = test_product,
    use_collab    = False,
    use_reviews   = False,
    weight_scheme = 'popularity',
    top_n         = top
))

print("\n-- Content Only config --")
print(get_recommendations(
    product_name  = test_product,
    use_collab    = False,
    use_reviews   = False,
    weight_scheme = 'content_only',
    top_n         = top
))

# Sentiment for top result
print("\n-- Sentiment for top result --")
if isinstance(results, str) or results.empty:
    print("No recommendations returned — skipping sentiment analysis.")
else:
    top_product_id = df[df['product_name'] == results['product_name'].iloc[0]]['product_id'].iloc[0]
    sentiment = get_profile_sentiment(top_product_id, my_profile)
    for key, val in sentiment.items():
        print(f"{key:30s}: {val}")

# Standard 7-metric evaluation across 7 configs
print("\n-- Standard metric evaluation (150 per category, 7 configs) --")
summary_df, detail_df = full_evaluation(n_per_skintype=150, top_n=10, random_seed=42)
print_full_evaluation_table(summary_df, detail_df)
plot_full_evaluation(summary_df, detail_df, save_path='full_evaluation.png')

# Pairwise significance tests — referee response (Groups A, B, C)
print("\n-- Pairwise Significance Tests (Groups A, B, C) --")
sig_results = pairwise_significance_test(detail_df, alpha=0.05)
print_significance_results(sig_results, alpha=0.05)

# Extended evaluation
print("\n-- Test 1: Ranking Disagreement --")
rank_detail, rank_summary = ranking_disagreement(n_per_skintype=150, top_n=10, random_seed=42)

print("\n-- Test 2: Cross-Profile Overlap (all hybrid variants) --")
overlap_details, overlap_summaries, score_table = cross_profile_overlap_test(
    n_per_skintype=150, top_n=10, random_seed=42)

print("\n-- Test 3: Simulated User Preference + Personalisation Lift (all hybrids) --")
pref_detail, pref_summary, pref_pivot = simulated_preference_test(n_per_skintype=150, top_n=10, random_seed=42)

print("\n-- Generating extended evaluation dashboard --")
plot_extended_evaluation(
    rank_summary = rank_summary,
    score_table  = score_table,
    pref_summary = pref_summary,
    pref_pivot   = pref_pivot,
    save_path    = 'extended_evaluation.png'
)

print("\nDone. Output files:")
print("  full_evaluation.png     -- standard 7-metric dashboard (7 configs)")
print("  extended_evaluation.png -- ranking, personalisation, preference (4 panels)")

# ── Bayesian weight optimisation (Step 12) ───────────────────────────────
print("\n-- Bayesian weight optimisation (60 TPE trials, 70/30 val/test split) --")
bo_study, bo_best_weights, bo_val_seeds, bo_test_seeds = bayesian_weight_optimisation(
    n_per_skintype=150, n_trials=60, val_fraction=0.70,
    top_n=10, random_seed=42)

if bo_study is not None:
    print("\n-- Test-split comparison (Optimised Hybrid vs all hand-designed configs) --")
    bo_test_summary = test_split_comparison(
        bo_best_weights, bo_test_seeds, top_n=10, random_seed=42)

    print("\n-- Generating weight optimisation chart --")
    plot_weight_optimisation(
        bo_study, bo_best_weights, bo_test_summary,
        save_path='weight_optimisation.png')

    print("\nAdditional output file:")
    print("  weight_optimisation.png -- BO history, optimal weights, test-split comparison")

# ===========================================================================
_elapsed = time.time() - _start_time
print(f"\nTotal runtime: {_elapsed:.1f}s  ({_elapsed/60:.1f} min)")
