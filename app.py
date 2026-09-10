
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse


# =============================================================================
# APPLICATION CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Cultural Recommendation Prototype",
    page_icon="🌍",
    layout="wide",
)


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

PRODUCT_FILE = DATA_DIR / "dynamic_product_metadata.parquet"
PROFILE_FILE = DATA_DIR / "synthetic_user_profiles.csv"

TFIDF_MATRIX_FILE = (
    MODEL_DIR
    / "phase_m4_3_1_tfidf_product_matrix.npz"
)

TFIDF_VOCAB_FILE = (
    MODEL_DIR
    / "m4_3_1_tfidf_vocabulary.json"
)

TFIDF_IDF_FILE = (
    MODEL_DIR
    / "m4_3_1_tfidf_idf.npy"
)

PROFILE_CBF_FILE = (
    MODEL_DIR
    / "m4_3_2_profile_cbf_matrix.npy"
)

CONFIG_FILE = (
    MODEL_DIR
    / "h17_6_step3b_aligned_configuration.json"
)


# =============================================================================
# FROZEN RESEARCH CONFIGURATION
# =============================================================================

CONTENT_WEIGHT = 0.20
KNN_WEIGHT = 0.20
CULTURAL_WEIGHT = 0.60
K = 5

PROFILE_WEIGHTS = {
    "preferred_regions": 0.25,
    "preferred_ethnic_groups": 0.15,
    "preferred_categories": 0.25,
    "preferred_festivals": 0.15,
    "preferred_traditions": 0.15,
    "preferred_products": 0.05,
}

PROFILE_DIMENSIONS = list(
    PROFILE_WEIGHTS.keys()
)


# =============================================================================
# CACHED DATA LOADING
# =============================================================================

@st.cache_resource
def load_engine():

    # -------------------------------------------------------------------------
    # Product metadata
    # -------------------------------------------------------------------------

    products = pd.read_parquet(
        PRODUCT_FILE
    ).copy()

    products["parent_asin"] = (
        products["parent_asin"]
        .astype(str)
    )

    # -------------------------------------------------------------------------
    # Synthetic profiles
    # -------------------------------------------------------------------------

    profiles = pd.read_csv(
        PROFILE_FILE
    ).copy()

    profiles["user_id"] = (
        profiles["user_id"]
        .astype(str)
    )

    # -------------------------------------------------------------------------
    # Exact M4.3.1 TF-IDF representation
    # -------------------------------------------------------------------------

    tfidf_matrix = sparse.load_npz(
        TFIDF_MATRIX_FILE
    )

    with open(
        TFIDF_VOCAB_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        vocabulary = json.load(f)

    vocabulary = {
        str(k): int(v)
        for k, v in vocabulary.items()
    }

    idf = np.asarray(
        np.load(
            TFIDF_IDF_FILE
        ),
        dtype=np.float32
    )

    # -------------------------------------------------------------------------
    # Exact M4.3.2 profile CBF representation
    # -------------------------------------------------------------------------

    profile_cbf = np.load(
        PROFILE_CBF_FILE
    )

    # -------------------------------------------------------------------------
    # Frozen component normalisation.
    #
    # These maxima are the maxima recorded by the frozen M4.3.3 hybrid
    # research output.
    # -------------------------------------------------------------------------

    content_max = 0.23977863788604736
    knn_max = 0.23362524807453156
    cultural_max = 0.4

    return {
        "products": products,
        "profiles": profiles,
        "tfidf_matrix": tfidf_matrix,
        "vocabulary": vocabulary,
        "idf": idf,
        "profile_cbf": profile_cbf,
        "content_max": content_max,
        "knn_max": knn_max,
        "cultural_max": cultural_max,
    }


# =============================================================================
# LOAD ENGINE
# =============================================================================

try:

    ENGINE = load_engine()

except Exception as exc:

    st.error(
        "The recommendation engine could not be loaded."
    )

    st.exception(
        exc
    )

    st.stop()


products = ENGINE["products"]
profiles = ENGINE["profiles"]

tfidf_matrix = ENGINE["tfidf_matrix"]
vocabulary = ENGINE["vocabulary"]
idf = ENGINE["idf"]

profile_cbf = ENGINE["profile_cbf"]

content_max = ENGINE["content_max"]
knn_max = ENGINE["knn_max"]
cultural_max = ENGINE["cultural_max"]


# =============================================================================
# PREFERENCE PARSING
# =============================================================================

def parse_preferences(value):

    if value is None:

        return set()

    if pd.isna(value):

        return set()

    text = str(
        value
    ).strip()

    if not text:

        return set()

    text = text.strip(
        "[](){}"
    )

    parts = re.split(
        r"\s*[;,|]\s*",
        text
    )

    output = set()

    for part in parts:

        cleaned = (
            part
            .strip()
            .strip("'\"")
            .strip()
            .lower()
        )

        if cleaned:

            output.add(
                cleaned
            )

    return output


def build_preferences(
    regions,
    ethnic_groups,
    categories,
    festivals,
    traditions,
    product_interest,
):

    return {

        "preferred_regions":
            {
                str(x).strip().lower()
                for x in regions
                if str(x).strip()
            },

        "preferred_ethnic_groups":
            {
                str(x).strip().lower()
                for x in ethnic_groups
                if str(x).strip()
            },

        "preferred_categories":
            {
                str(x).strip().lower()
                for x in categories
                if str(x).strip()
            },

        "preferred_festivals":
            {
                str(x).strip().lower()
                for x in festivals
                if str(x).strip()
            },

        "preferred_traditions":
            {
                str(x).strip().lower()
                for x in traditions
                if str(x).strip()
            },

        "preferred_products":
            {
                token.lower()
                for token in re.findall(
                    r"[a-z0-9]+",
                    str(product_interest).lower()
                )
                if token
            },
    }


# =============================================================================
# SYNTHETIC PROFILE REPRESENTATIONS
# =============================================================================

def get_profile_preferences(row):

    output = {}

    for dimension in PROFILE_DIMENSIONS:

        output[dimension] = (
            parse_preferences(
                row.get(
                    dimension,
                    ""
                )
            )
        )

    return output


SYNTHETIC_PREFERENCES = {}

for _, row in profiles.iterrows():

    uid = str(
        row["user_id"]
    )

    SYNTHETIC_PREFERENCES[uid] = (
        get_profile_preferences(row)
    )


# =============================================================================
# PROFILE SIMILARITY
# =============================================================================

def profile_feature_dictionary(
    preferences
):

    features = {}

    for dimension in PROFILE_DIMENSIONS:

        weight = PROFILE_WEIGHTS[
            dimension
        ]

        for value in preferences.get(
            dimension,
            set()
        ):

            key = (
                dimension
                + "::"
                + str(value).strip().lower()
            )

            features[key] = weight

    return features


def cosine_profile_similarity(
    preferences_a,
    preferences_b
):

    a = profile_feature_dictionary(
        preferences_a
    )

    b = profile_feature_dictionary(
        preferences_b
    )

    if not a or not b:

        return 0.0

    keys = set(a) | set(b)

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for key in keys:

        av = float(
            a.get(
                key,
                0.0
            )
        )

        bv = float(
            b.get(
                key,
                0.0
            )
        )

        dot += av * bv
        norm_a += av * av
        norm_b += bv * bv

    denominator = (
        np.sqrt(norm_a)
        * np.sqrt(norm_b)
    )

    if denominator <= 0:

        return 0.0

    return float(
        np.clip(
            dot / denominator,
            0.0,
            1.0
        )
    )


# =============================================================================
# ONTOLOGY VALUE REPRESENTATION
# =============================================================================

@st.cache_resource
def prepare_ontology():

    ontology_arrays = {}

    ontology_columns = [
        "cultural_region",
        "ethnic_group",
        "cultural_category",
        "festival_relevance",
        "traditional_significance",
    ]

    for column in ontology_columns:

        ontology_arrays[column] = [

            parse_preferences(value)

            for value in products[column]
        ]

    # Product-content tokens for preferred-product matching.

    content_tokens = []

    for value in products[
        "product_content_text"
    ].fillna("").astype(str):

        content_tokens.append(
            set(
                re.findall(
                    r"[a-z0-9]+",
                    value.lower()
                )
            )
        )

    return (
        ontology_arrays,
        content_tokens
    )


ONTOLOGY_ARRAYS, PRODUCT_CONTENT_TOKENS = (
    prepare_ontology()
)


# =============================================================================
# DYNAMIC CULTURAL AFFINITY
# =============================================================================

def calculate_cultural_affinity(
    preferences
):

    n = len(products)

    scores = np.zeros(
        n,
        dtype=np.float32
    )

    mapping = [
        (
            "preferred_regions",
            "cultural_region",
            0.25,
        ),
        (
            "preferred_ethnic_groups",
            "ethnic_group",
            0.15,
        ),
        (
            "preferred_categories",
            "cultural_category",
            0.25,
        ),
        (
            "preferred_festivals",
            "festival_relevance",
            0.15,
        ),
        (
            "preferred_traditions",
            "traditional_significance",
            0.15,
        ),
    ]

    for (
        preference_dimension,
        ontology_dimension,
        weight,
    ) in mapping:

        preferred = preferences.get(
            preference_dimension,
            set()
        )

        if not preferred:

            continue

        values = ONTOLOGY_ARRAYS[
            ontology_dimension
        ]

        for i, product_values in enumerate(
            values
        ):

            if product_values & preferred:

                scores[i] += np.float32(
                    weight
                )

    # Preferred-product signal.

    preferred_products = preferences.get(
        "preferred_products",
        set()
    )

    if preferred_products:

        for i, tokens in enumerate(
            PRODUCT_CONTENT_TOKENS
        ):

            if tokens & preferred_products:

                scores[i] += np.float32(
                    0.05
                )

    return np.clip(
        scores,
        0.0,
        1.0
    ).astype(
        np.float32
    )


# =============================================================================
# DYNAMIC CONTENT SCORE
# =============================================================================

def dynamic_content_score(
    product_interest
):

    query = str(
        product_interest or ""
    ).strip().lower()

    if not query:

        return np.zeros(
            len(products),
            dtype=np.float32
        )

    raw_tokens = re.findall(
        r"(?u)\b\w\w+\b",
        query
    )

    stop_words = {
        "a", "an", "and", "are", "as",
        "at", "be", "by", "for", "from",
        "has", "have", "he", "in", "is",
        "it", "its", "of", "on", "or",
        "that", "the", "this", "to",
        "was", "were", "will", "with",
        "you", "your"
    }

    tokens = [
        token
        for token in raw_tokens
        if token not in stop_words
    ]

    features = list(
        tokens
    )

    for i in range(
        len(tokens) - 1
    ):

        features.append(
            tokens[i]
            + " "
            + tokens[i + 1]
        )

    counts = {}

    for feature in features:

        if feature in vocabulary:

            index = vocabulary[
                feature
            ]

            counts[index] = (
                counts.get(
                    index,
                    0.0
                )
                + 1.0
            )

    if not counts:

        return np.zeros(
            len(products),
            dtype=np.float32
        )

    indices = np.asarray(
        list(counts.keys()),
        dtype=np.int32
    )

    values = np.asarray(
        list(counts.values()),
        dtype=np.float32
    )

    row = np.zeros(
        len(indices),
        dtype=np.int32
    )

    query_tf = sparse.csr_matrix(
        (
            values,
            (
                row,
                indices
            )
        ),
        shape=(
            1,
            len(vocabulary)
        ),
        dtype=np.float32
    )

    query_tfidf = query_tf.multiply(
        idf
    )

    norm = float(
        np.sqrt(
            query_tfidf.multiply(
                query_tfidf
            ).sum()
        )
    )

    if norm <= 0:

        return np.zeros(
            len(products),
            dtype=np.float32
        )

    query_tfidf = (
        query_tfidf
        / norm
    )

    scores = (
        tfidf_matrix
        @ query_tfidf.T
    ).toarray().ravel()

    return scores.astype(
        np.float32
    )


# =============================================================================
# DYNAMIC PROFILE-BASED KNN
# =============================================================================

def dynamic_knn_score(
    participant_preferences
):

    similarities = np.zeros(
        len(profiles),
        dtype=np.float32
    )

    for i, uid in enumerate(
        profiles["user_id"]
    ):

        similarities[i] = (
            cosine_profile_similarity(
                participant_preferences,
                SYNTHETIC_PREFERENCES[
                    str(uid)
                ]
            )
        )

    positive = np.where(
        similarities > 0
    )[0]

    if len(positive) == 0:

        return (
            np.zeros(
                len(products),
                dtype=np.float32
            ),
            similarities,
            []
        )

    ordered = positive[
        np.argsort(
            -similarities[
                positive
            ],
            kind="stable"
        )
    ]

    neighbours = ordered[
        :K
    ]

    weights = similarities[
        neighbours
    ]

    denominator = float(
        weights.sum()
    )

    if denominator <= 0:

        return (
            np.zeros(
                len(products),
                dtype=np.float32
            ),
            similarities,
            []
        )

    transferred = (
        weights[:, None]
        * profile_cbf[
            neighbours
        ]
    ).sum(
        axis=0
    )

    transferred /= denominator

    return (
        transferred.astype(
            np.float32
        ),
        similarities,
        neighbours.tolist()
    )


# =============================================================================
# COMPLETE DYNAMIC RECOMMENDATION ENGINE
# =============================================================================

def generate_recommendations(
    participant_preferences,
    product_interest,
    top_n=10,
):

    # Dynamic content.

    content_scores = (
        dynamic_content_score(
            product_interest
        )
    )

    # Dynamic profile-based KNN.

    (
        knn_scores,
        profile_similarities,
        neighbours,
    ) = dynamic_knn_score(
        participant_preferences
    )

    # Dynamic cultural affinity.

    cultural_scores = (
        calculate_cultural_affinity(
            participant_preferences
        )
    )

    # Frozen global maximum normalisation.

    content_norm = (
        content_scores
        / np.float32(
            content_max
        )
    )

    knn_norm = (
        knn_scores
        / np.float32(
            knn_max
        )
    )

    cultural_norm = (
        cultural_scores
        / np.float32(
            cultural_max
        )
    )

    # Frozen hybrid.

    hybrid_scores = (

        np.float32(
            CONTENT_WEIGHT
        )
        * content_norm

        +

        np.float32(
            KNN_WEIGHT
        )
        * knn_norm

        +

        np.float32(
            CULTURAL_WEIGHT
        )
        * cultural_norm
    )

    # Cultural eligibility gate.

    eligible = (
        products[
            "cultural_recommendation_eligible"
        ]
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )

    hybrid_scores[
        ~eligible
    ] = 0.0

    # Stable ranking.

    order = np.argsort(
        -hybrid_scores,
        kind="stable"
    )

    order = [
        i
        for i in order
        if hybrid_scores[i] > 0
    ][:top_n]

    result = products.iloc[
        order
    ].copy()

    result[
        "content_similarity"
    ] = content_scores[
        order
    ]

    result[
        "knn_collaborative_score"
    ] = knn_scores[
        order
    ]

    result[
        "cultural_similarity"
    ] = cultural_scores[
        order
    ]

    result[
        "hybrid_score"
    ] = hybrid_scores[
        order
    ]

    result[
        "recommendation_rank"
    ] = np.arange(
        1,
        len(result) + 1
    )

    return (
        result,
        profile_similarities,
        neighbours,
    )


# =============================================================================
# UI
# =============================================================================

st.title(
    "🌍 AI-Driven Cultural Recommendation System"
)

st.caption(
    "Research prototype — dynamic participant-driven recommendation"
)

st.info(
    "Your selected cultural preferences and product interest directly "
    "influence the recommendation ranking. The prototype uses the frozen "
    "research configuration: 20% content, 20% profile-based KNN and "
    "60% cultural affinity."
)


# =============================================================================
# PREFERENCE INPUT
# =============================================================================

st.header(
    "1. Tell us what you are interested in"
)

regions = sorted(
    {
        value
        for values in ONTOLOGY_ARRAYS[
            "cultural_region"
        ]
        for value in values
        if value
    }
)

ethnic_groups = sorted(
    {
        value
        for values in ONTOLOGY_ARRAYS[
            "ethnic_group"
        ]
        for value in values
        if value
    }
)

categories = sorted(
    {
        value
        for values in ONTOLOGY_ARRAYS[
            "cultural_category"
        ]
        for value in values
        if value
    }
)

festivals = sorted(
    {
        value
        for values in ONTOLOGY_ARRAYS[
            "festival_relevance"
        ]
        for value in values
        if value
    }
)

traditions = sorted(
    {
        value
        for values in ONTOLOGY_ARRAYS[
            "traditional_significance"
        ]
        for value in values
        if value
    }
)


col1, col2 = st.columns(2)

with col1:

    selected_regions = st.multiselect(
        "Cultural Region",
        regions,
        key="regions",
    )

    selected_ethnic_groups = st.multiselect(
        "Ethnic Group",
        ethnic_groups,
        key="ethnic_groups",
    )

    selected_categories = st.multiselect(
        "Cultural Category",
        categories,
        key="categories",
    )


with col2:

    selected_festivals = st.multiselect(
        "Festival Relevance",
        festivals,
        key="festivals",
    )

    selected_traditions = st.multiselect(
        "Traditional Significance",
        traditions,
        key="traditions",
    )

    product_interest = st.text_input(
        "Product Interest",
        placeholder=(
            "e.g. Kente, Suya, Biryani, "
            "African print, Chinese New Year decorations"
        ),
        key="product_interest",
    )


# =============================================================================
# PREFERENCE SUMMARY
# =============================================================================

participant_preferences = build_preferences(
    selected_regions,
    selected_ethnic_groups,
    selected_categories,
    selected_festivals,
    selected_traditions,
    product_interest,
)

st.subheader(
    "Preference Summary"
)

summary_items = []

for label, dimension in [
    ("Cultural Region", "preferred_regions"),
    ("Ethnic Group", "preferred_ethnic_groups"),
    ("Cultural Category", "preferred_categories"),
    ("Festival", "preferred_festivals"),
    ("Traditional Significance", "preferred_traditions"),
    ("Product Interest", "preferred_products"),
]:

    values = participant_preferences[
        dimension
    ]

    summary_items.append(
        f"**{label}:** "
        + (
            ", ".join(
                sorted(values)
            )
            if values
            else "Not specified"
        )
    )

st.markdown(
    "  \n".join(
        summary_items
    )
)


# =============================================================================
# GENERATE
# =============================================================================

st.header(
    "2. Generate Recommendations"
)

generate = st.button(
    "Generate Recommendations",
    type="primary",
    use_container_width=True,
)


if generate:

    if not any(
        participant_preferences.values()
    ):

        st.warning(
            "Please select at least one cultural preference "
            "or enter a product interest."
        )

        st.stop()

    with st.spinner(
        "Calculating your personalised recommendations..."
    ):

        (
            recommendations,
            profile_similarities,
            neighbours,
        ) = generate_recommendations(
            participant_preferences,
            product_interest,
            top_n=10,
        )

    st.session_state[
        "recommendations"
    ] = recommendations

    st.session_state[
        "profile_similarities"
    ] = profile_similarities

    st.session_state[
        "neighbours"
    ] = neighbours


# =============================================================================
# RESULTS
# =============================================================================

if (
    "recommendations"
    in st.session_state
):

    recommendations = st.session_state[
        "recommendations"
    ]

    profile_similarities = st.session_state[
        "profile_similarities"
    ]

    neighbours = st.session_state[
        "neighbours"
    ]

    st.header(
        "3. Your Recommendations"
    )

    if recommendations.empty:

        st.warning(
            "No recommendations were generated for these preferences."
        )

    else:

        st.success(
            "The recommendations below were calculated dynamically "
            "from your current preferences."
        )

        # ---------------------------------------------------------------------
        # Model diagnostics
        # ---------------------------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Recommendations",
                len(recommendations)
            )

        with col2:

            st.metric(
                "Culturally Eligible",
                f"{recommendations['cultural_recommendation_eligible'].mean() * 100:.0f}%"
            )

        with col3:

            st.metric(
                "Mean Cultural Affinity",
                f"{recommendations['cultural_similarity'].mean():.3f}"
            )

        with col4:

            st.metric(
                "Mean Hybrid Score",
                f"{recommendations['hybrid_score'].mean():.3f}"
            )


        # ---------------------------------------------------------------------
        # Profile similarity diagnostics
        # ---------------------------------------------------------------------

        best_index = int(
            np.argmax(
                profile_similarities
            )
        )

        best_profile = str(
            profiles.iloc[
                best_index
            ]["user_id"]
        )

        st.caption(
            "The profile-based KNN component uses the most similar "
            "validated synthetic profile representations as preference "
            "transfer neighbours. No behavioural user-interaction data "
            "are used."
        )

        with st.expander(
            "View recommendation-engine diagnostics"
        ):

            st.write(
                "Closest validated profile:",
                best_profile
            )

            st.write(
                "Selected KNN neighbours:",
                [
                    str(
                        profiles.iloc[i][
                            "user_id"
                        ]
                    )
                    for i in neighbours
                ]
            )

            st.write(
                "Frozen configuration:"
            )

            st.write(
                {
                    "Content": CONTENT_WEIGHT,
                    "Profile-Based KNN": KNN_WEIGHT,
                    "Cultural": CULTURAL_WEIGHT,
                    "K": K,
                }
            )


        # ---------------------------------------------------------------------
        # Recommendation cards
        # ---------------------------------------------------------------------

        for _, row in recommendations.iterrows():

            rank = int(
                row[
                    "recommendation_rank"
                ]
            )

            title = str(
                row.get(
                    "title",
                    "Untitled product"
                )
            )

            st.subheader(
                f"{rank}. {title}"
            )

            col1, col2 = st.columns(
                [3, 1]
            )

            with col1:

                description = str(
                    row.get(
                        "description",
                        ""
                    )
                )

                if description:

                    st.write(
                        description[:800]
                    )

                st.write(
                    f"**Cultural Region:** "
                    f"{row.get('cultural_region', 'Not available')}"
                )

                st.write(
                    f"**Ethnic Group:** "
                    f"{row.get('ethnic_group', 'Not available')}"
                )

                st.write(
                    f"**Cultural Category:** "
                    f"{row.get('cultural_category', 'Not available')}"
                )

                st.write(
                    f"**Festival Relevance:** "
                    f"{row.get('festival_relevance', 'Not available')}"
                )

                st.write(
                    f"**Traditional Significance:** "
                    f"{row.get('traditional_significance', 'Not available')}"
                )

            with col2:

                st.metric(
                    "Hybrid Score",
                    f"{row['hybrid_score']:.3f}"
                )

                st.metric(
                    "Cultural Affinity",
                    f"{row['cultural_similarity']:.3f}"
                )

                st.metric(
                    "Content Similarity",
                    f"{row['content_similarity']:.3f}"
                )

                st.metric(
                    "KNN Preference",
                    f"{row['knn_collaborative_score']:.3f}"
                )

                if bool(
                    row[
                        "cultural_recommendation_eligible"
                    ]
                ):

                    st.success(
                        "Culturally eligible"
                    )

                else:

                    st.warning(
                        "Cultural evidence insufficient"
                    )

            # -----------------------------------------------------------------
            # Explanation
            # -----------------------------------------------------------------

            explanation_parts = []

            if (
                float(
                    row[
                        "cultural_similarity"
                    ]
                ) > 0
            ):

                explanation_parts.append(
                    "matches one or more selected cultural preferences"
                )

            if (
                float(
                    row[
                        "content_similarity"
                    ]
                ) > 0
            ):

                explanation_parts.append(
                    "has content similarity to the stated product interest"
                )

            if (
                float(
                    row[
                        "knn_collaborative_score"
                    ]
                ) > 0
            ):

                explanation_parts.append(
                    "receives preference-transfer support from similar "
                    "validated profile representations"
                )

            if explanation_parts:

                explanation = (
                    "This recommendation "
                    + "; ".join(
                        explanation_parts
                    )
                    + "."
                )

            else:

                explanation = (
                    "This product passed the recommendation scoring "
                    "and cultural eligibility process."
                )

            st.info(
                "**Why this product was recommended:** "
                + explanation
            )

            st.divider()


# =============================================================================
# RESEARCH PARTICIPATION
# =============================================================================

st.header(
    "4. Research Questionnaire"
)

st.write(
    "After exploring the recommendations, please complete the "
    "research questionnaire provided by the researcher. "
    "Your responses will be used to evaluate perceived cultural "
    "relevance, personalisation, product discovery and usefulness."
)

st.caption(
    "Please evaluate the recommendations based on your own experience. "
    "The prototype does not identify which model component produced "
    "a recommendation and does not indicate an expected outcome."
)


# =============================================================================
# RESEARCH CONFIGURATION
# =============================================================================

with st.expander(
    "Research model configuration"
):

    st.write(
        "**Frozen configuration**"
    )

    st.write(
        {
            "Content-Based Filtering": "0.20",
            "Profile-Based KNN": "0.20",
            "Cultural Affinity": "0.60",
            "K": "5",
        }
    )

    st.write(
        "The prototype performs dynamic participant-driven "
        "recommendation calculation. No model retraining or "
        "participant-response-based tuning occurs during use."
    )
