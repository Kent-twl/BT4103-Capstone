import streamlit as st
import datetime


# ============================================================
# Configuration and Initialisation
# ============================================================

# Confiugure the app to have a wide layout
st.set_page_config(
    layout="wide",
    # initial_sidebar_state="collapsed"
)

# Initialize selections as an empty list in session state
if "selections" not in st.session_state:
    st.session_state.selections = []

# Initialize date as None in session state
if "date" not in st.session_state:
    st.session_state.date = None


# Selected options in multiselect will have their text shown in full
def update_multiselect_style():
    st.markdown(
        """
        <style>
            .stMultiSelect [data-baseweb="tag"] {
                height: fit-content;
            }
            .stMultiSelect [data-baseweb="tag"] span[title] {
                white-space: normal; max-width: 100%; overflow-wrap: anywhere;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


update_multiselect_style()


# ============================================================
# Adding Pages and Routing
# ============================================================


# Define the log in page for user to select the date and functionalities
def login():
    SELECTIONS = [
        "Business Intelligence",
        "Anomaly Detection",
        "ASIC Reporting",
        "<Future Functionality 1>",
        "<Future Functionality 2>",
    ]
    st.header("IRESS End of Day Reporting Dashboard View Customization")
    today = datetime.datetime.now()
    date = st.date_input(
        "Which date's data would you like to view?", max_value=today, value=None
    )
    selections = st.multiselect(
        "Which functinalities and charts would you like to view?",
        SELECTIONS,
        placeholder="Nothing selected yet",
    )
    if st.button("Confirm", icon=":material/arrow_right_alt:"):
        st.session_state.selections = selections
        st.session_state.date = date
        st.rerun()


# Define the log out page and clear the session_state
def logout():
    st.session_state.selections = []
    st.session_state.date = None
    # Reset the chatbot
    st.cache_resource.clear()
    st.session_state.messages = []
    for key in st.session_state.bi_figures_changed:
        st.session_state.bi_figures_changed[key] = True
    st.rerun()


# Retrieve current session_state
selections = st.session_state.selections
date = st.session_state.date

# Add logout page to st.Page
logout_page = st.Page(
    logout,
    title="Return to Chart Selections",
    icon=":material/logout:",
    default=(not bool(selections) or not bool(selections)),
)

# Add dashboard page to st.Page
dashboard_page = st.Page(
    "dashboard.py",
    title="Current Dashboard",
    icon=":material/visibility:",
    default=(bool(selections) and bool(date)),
)

# Add chatbot page to st.Page
chatbot_page = st.Page(
    "../llm/chatbot/st_integration2.py",
    title="Ask Chatbot Ad-Hoc Queries",
    icon=":material/robot_2:",
    default=False,
)

# Routing between different pages
# Only display the dashboard and chatbot pages when selections for dates and charts are made
if selections and (date is not None):
    # Display logout option if selections are made
    pg = st.navigation(
        {
            "": [
                dashboard_page,
                chatbot_page,
                logout_page,
            ]
        }
    )
else:
    # Display login if no selections
    pg = st.navigation([st.Page(login)])

# Run the navigation
pg.run()
