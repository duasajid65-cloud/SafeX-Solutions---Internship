import pandas as pd
import plotly.express as px
import streamlit as st

#
# Page Configuration
#
st.set_page_config(
    page_title = "SafeX Internship Cohort Dashboard", page_icon = "📊", layout = "wide")


@st.cache_data
def load_data():
    return pd.read_csv("Internship Cohort Dashboard.csv", keep_default_na = False)

df = load_data()

#
# Title
#
st.title("📊 SafeX Internship Cohort Data Analysis Dashboard")

st.markdown(
    """
    **Purpose:** Analyze the composition of the SafeX Internship Cohort using privacy-safe, aggregated information.

    Personal identifiers such as names, emails are intentionally excluded.
    """
)

st.divider()    

#
# Sidebar Filters
#
st.sidebar.header("Dashboard Filters")

# Field Filter

field_options = sorted(df["Field"].dropna().unique())

selected_fields = st.sidebar.multiselect("Select Field", field_options, default = field_options)


# University Filter

university_options = sorted(df["University"].dropna().unique())

selected_universities = st.sidebar.multiselect("Select University", university_options)


# Gender Filter

gender_options = sorted(df["Gender"].dropna().unique())

selected_gender = st.sidebar.multiselect("Select Gender", gender_options, default = gender_options)


# Semester Filter

semester_options = sorted(df["Semester"].dropna().unique())

selected_semesters = st.sidebar.multiselect("Select Academic Stage", semester_options)

#
# Apply Filters
#
filtered_df = df[
    df["Field"].isin(selected_fields)
    & df["Gender"].isin(selected_gender)].copy()

if selected_universities:
    filtered_df = filtered_df[
        filtered_df["University"].isin(selected_universities)]

if selected_semesters:
    filtered_df = filtered_df[
        filtered_df["Semester"].isin(selected_semesters)]
    
#
# KPI Metrics
#
st.subheader("📌 Cohort Overview")

col1, col2, col3, col4 = st.columns(4)

total_interns = len(filtered_df)
total_fields = filtered_df["Field"].nunique()
total_universities = filtered_df["University"].nunique()
total_groups = filtered_df["Group"].nunique()


col1.metric("Total Interns", total_interns)
col2.metric("Fields", total_fields)
col3.metric("Universities", total_universities)
col4.metric("Groups", total_groups)

st.divider()

#
# Visualization
#
# 1. Interns By Field
#

st.subheader("1️⃣. Interns by Field")

field_data = (filtered_df["Field"].value_counts().reset_index())

field_data.columns = ["Field", "Interns"]

field_data = field_data.sort_values("Interns", ascending = True)


fig_field = px.bar(
    field_data, 
    x = "Interns", 
    y = "Field", 
    orientation = "h", 
    text = "Interns", 
    title = "Number of Fields by Interns"
)     

fig_field.update_layout(
    height = 500,
    xaxis_title = "Number of Interns",
    yaxis_title = "Field"
)

st.plotly_chart(fig_field, use_container_width = True)

#
# 2. Top Universities
#
st.subheader("2️⃣. Top Universities")

university_data = (filtered_df["University"].value_counts().head(15).reset_index())

university_data.columns = ["University", "Interns"]

university_data = university_data.sort_values("Interns", ascending = True)

fig_university = px.bar(
    university_data,
    x = "Interns",
    y = "University",
    orientation = "h",
    text = "Interns",
    title = "Top 15 Universities"
)

fig_university.update_layout(
    height = 500,
    xaxis_title = "Number of Interns",
    yaxis_title = "University"
)

st.plotly_chart(fig_university, use_container_width = True)

#
# 3. Gender Distribution
#
st.subheader("3️⃣. Gender Distribution")

gender_data = (filtered_df["Gender"].value_counts().reset_index())

gender_data.columns = ["Gender", "Interns"]


fig_gender = px.pie(
    gender_data,
    names = "Gender",
    values = "Interns",
    hole = 0.4,
    title = "Gender Distribution"
)

st.plotly_chart(fig_gender, use_container_width = True)

#
# 4. Group Size Distribution
#
st.subheader("4️⃣. Group Size Distribution")

group_sizes = (filtered_df.groupby("Group").size().reset_index(name = "Group Size"))

group_distribution = (group_sizes["Group Size"].value_counts().sort_index().reset_index())

group_distribution.columns = ["Group Size", "Number of Groups"]



fig_groups = px.bar(
    group_distribution,
    x = "Group Size",
    y = "Number of Groups",
    text = "Number of Groups",
    title = "Distribution of Group Sizes"
)

fig_groups.update_layout(
    height = 500,
    xaxis_title = "Number of Interns in Group",
    yaxis_title = "Number of Groups"
)

st.plotly_chart(fig_groups, use_container_width = True)

#
# 5. Gender by Field Heatmap
#
st.subheader("5️⃣. Gender Distribution by Field")

gender_field = pd.crosstab(filtered_df["Field"], filtered_df["Gender"])


fig_heatmap = px.imshow(
    gender_field,
    text_auto = True,
    aspect = "auto",
    title = "Gender Distribution across Internship Fields",
    labels = {
        "x": "Gender",
        "y": "Field",
        "color": "Number of Interns"
    }
)
 
st.plotly_chart(fig_heatmap, use_container_width = True)

#
# 6. Semester / Academic Stage
#
st.subheader("6️⃣. Academic Stage Distribution")

semester_clean = filtered_df["Semester"].str.strip()

semester_data = (semester_clean.value_counts().reset_index())

semester_data.columns = ["Academic Stage", "Interns"]

semester_data["num"] = semester_data["Academic Stage"].str.extract(r'(\d+)').astype(float)


semester_data["is_completed"] = semester_data["Academic Stage"].str.contains("completed").astype(int)

semester_data = semester_data.sort_values(["num", "is_completed"])


fig_semester = px.bar(
    semester_data,
    x = "Academic Stage",
    y = "Interns", 
    text = "Interns",
    title = "Interns by Academic Stage"
)

fig_semester.update_layout(
    xaxis_tickangle = 90
)

st.plotly_chart(fig_semester, use_container_width = True)

#
# Key Insights
# 
st.divider()

st.subheader("💡 Key Insights")

# Top Field

if not filtered_df.empty:
    
    top_field = (filtered_df["Field"].value_counts().idxmax())

    top_field_count = (filtered_df["Field"].value_counts().max())

    st.write(
    f". **{top_field}** is the largest internship field"
    f"with **{top_field_count} interns**."
    )
    
# Top University

if not filtered_df.empty:
    
    top_university = (filtered_df["University"].value_counts().idxmax())

    top_university_count = (filtered_df["University"].value_counts().max())

    st.write(
    f". **{top_university}** is the most represented university"
    f" contributing **{top_university_count} interns**."
    )
    
# Gender

gender_counts = filtered_df["Gender"].value_counts()

if not gender_counts.empty:

    gender_text = ", ".join(
        f"{gender}: {count}"
        for gender, count in gender_counts.items()
    )

    st.write(
        f". Gender Distribution: **{gender_text}**."
    )
    
# Group Size

if not group_sizes.empty:

    minimum_group = group_sizes["Group Size"].min()
    maximum_group = group_sizes["Group Size"].max()

    st.write(
        f". Group Sizes range from **{minimum_group} "
        f" to {maximum_group} interns**."
    )
        
#
# Privacy Notice
#
st.divider()

st.info(
    """
    **Privacy Notice**

    This dashboard uses only anonymized / aggregated cohort information. Names, emails, group leader details, and other personally
    identifiable information are not displayed.
    """
)

st.caption(
    "SafeX Internship Cohort Analysis | Python + Pandas + Plotly + Streamlit"
)       
           