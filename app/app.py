"""Simple transaction fraud-risk demo built with simulated training data."""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import demo_model  # noqa: E402


st.set_page_config(
    page_title="Transaction Risk Check",
    page_icon="💳",
    layout="centered",
)


@st.cache_resource
def load_model():
    return demo_model.train_demo_model()


model = load_model()

st.title("💳 Transaction Risk Check")
st.write(
    "Check a transaction or upload a small CSV to see which transactions "
    "may need a closer look."
)
st.warning(
    "Demo only: this model is trained on simulated transactions. "
    "Its predictions are not financial advice or proof of fraud."
)

with st.sidebar:
    st.header("Alert settings")
    threshold = st.slider(
        "Alert threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Lower thresholds flag more transactions for review.",
    )

single_tab, csv_tab = st.tabs(["Check one transaction", "Check a CSV"])

with single_tab:
    st.subheader("Transaction details")
    with st.form("transaction_form"):
        amount = st.number_input(
            "Amount",
            min_value=0.0,
            value=125.0,
            step=25.0,
            help="Amount in the transaction's currency.",
        )
        transaction_type = st.selectbox(
            "Transaction type", demo_model.TRANSACTION_TYPES
        )
        hour = st.slider("Hour of day", min_value=0, max_value=23, value=14)
        international = st.checkbox("Transaction is international")
        new_payee = st.checkbox("First transaction to this recipient")
        submitted = st.form_submit_button(
            "Check transaction", type="primary", use_container_width=True
        )

    if submitted:
        transaction = pd.DataFrame(
            [
                {
                    "montant": amount,
                    "type_transaction": transaction_type,
                    "heure": hour,
                    "international": international,
                    "nouveau_beneficiaire": new_payee,
                }
            ]
        )
        result = demo_model.score_transactions(
            model, transaction, threshold=threshold
        ).iloc[0]
        probability = result["probabilite_fraude"]

        st.metric("Estimated fraud risk", f"{probability:.1%}")
        if result["alerte"]:
            st.error("Alert: review this transaction.")
        else:
            st.success("No alert at the selected threshold.")

with csv_tab:
    st.subheader("Analyze a transaction batch")
    st.write("Use one row per transaction and these five columns:")
    st.code(
        "montant,type_transaction,heure,international,nouveau_beneficiaire\n"
        "125,paiement_carte,14,false,false\n"
        "2400,virement,2,true,true",
        language="csv",
    )
    st.caption(
        "Transaction types: "
        + ", ".join(demo_model.TRANSACTION_TYPES)
        + ". Boolean values: true/false, yes/no, or 1/0."
    )
    st.download_button(
        "Download example CSV",
        data=demo_model.example_transactions_csv(),
        file_name="transactions_demo.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="The CSV must contain the five columns shown above.",
    )
    if uploaded_file is not None:
        try:
            transactions = pd.read_csv(uploaded_file)
            results = demo_model.score_transactions(
                model, transactions, threshold=threshold
            )
        except (
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            UnicodeDecodeError,
        ) as exc:
            st.error(f"Could not read that CSV: {exc}")
        except ValueError as exc:
            st.error(f"Please check the CSV: {exc}")
        else:
            alerts = int(results["alerte"].sum())
            total = len(results)
            first_metric, second_metric = st.columns(2)
            first_metric.metric("Transactions checked", f"{total:,}")
            second_metric.metric("Alerts to review", f"{alerts:,}")

            st.dataframe(
                results.sort_values(
                    "probabilite_fraude", ascending=False
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="transactions_analysees.csv",
                mime="text/csv",
                key="download_results",
            )

st.divider()
st.caption(
    "This classroom prototype learns from generated examples. "
    "It is not trained on actual bank customers or transactions."
)
