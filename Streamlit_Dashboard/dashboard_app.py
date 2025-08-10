import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration and Data Generation ---
st.set_page_config(layout="wide", page_title="Air Purifier Market Intelligence Dashboard")

# Simulate Data for Demonstration
@st.cache_data
def generate_simulated_data():
    np.random.seed(42) # for reproducibility

    # 1. City Data for Market Prioritization
    cities = [
        {'City': 'Delhi', 'State': 'Delhi', 'Region': 'North', 'Population_Density': 20000, 'Average_Income_Lakhs': 15, 'AQI_Severity': 85},
        {'City': 'Mumbai', 'State': 'Maharashtra', 'Region': 'West', 'Population_Density': 21000, 'Average_Income_Lakhs': 18, 'AQI_Severity': 70},
        {'City': 'Kolkata', 'State': 'West Bengal', 'Region': 'East', 'Population_Density': 18000, 'Average_Income_Lakhs': 10, 'AQI_Severity': 75},
        {'City': 'Bengaluru', 'State': 'Karnataka', 'Region': 'South', 'Population_Density': 12000, 'Average_Income_Lakhs': 16, 'AQI_Severity': 65},
        {'City': 'Hyderabad', 'State': 'Telangana', 'Region': 'South', 'Population_Density': 10000, 'Average_Income_Lakhs': 14, 'AQI_Severity': 60},
        {'City': 'Chennai', 'State': 'Tamil Nadu', 'Region': 'South', 'Population_Density': 15000, 'Average_Income_Lakhs': 13, 'AQI_Severity': 55},
        {'City': 'Pune', 'State': 'Maharashtra', 'Region': 'West', 'Population_Density': 8000, 'Average_Income_Lakhs': 12, 'AQI_Severity': 50},
        {'City': 'Ahmedabad', 'State': 'Gujarat', 'Region': 'West', 'Population_Density': 7000, 'Average_Income_Lakhs': 11, 'AQI_Severity': 68},
        {'City': 'Lucknow', 'State': 'Uttar Pradesh', 'Region': 'North', 'Population_Density': 6000, 'Average_Income_Lakhs': 8, 'AQI_Severity': 80},
        {'City': 'Kanpur', 'State': 'Uttar Pradesh', 'Region': 'North', 'Population_Density': 5500, 'Average_Income_Lakhs': 7, 'AQI_Severity': 82},
        {'City': 'Jaipur', 'State': 'Rajasthan', 'Region': 'North', 'Population_Density': 5000, 'Average_Income_Lakhs': 9, 'AQI_Severity': 72},
        {'City': 'Surat', 'State': 'Gujarat', 'Region': 'West', 'Population_Density': 9000, 'Average_Income_Lakhs': 10, 'AQI_Severity': 63},
        {'City': 'Indore', 'State': 'Madhya Pradesh', 'Region': 'Central', 'Population_Density': 4500, 'Average_Income_Lakhs': 9, 'AQI_Severity': 67},
        {'City': 'Bhopal', 'State': 'Madhya Pradesh', 'Region': 'Central', 'Population_Density': 4000, 'Average_Income_Lakhs': 8, 'AQI_Severity': 65},
        {'City': 'Patna', 'State': 'Bihar', 'Region': 'East', 'Population_Density': 7000, 'Average_Income_Lakhs': 6, 'AQI_Severity': 90},
        {'City': 'Kochi', 'State': 'Kerala', 'Region': 'South', 'Population_Density': 3000, 'Average_Income_Lakhs': 11, 'AQI_Severity': 45},
        {'City': 'Chandigarh', 'State': 'Chandigarh', 'Region': 'North', 'Population_Density': 9000, 'Average_Income_Lakhs': 14, 'AQI_Severity': 58},
        {'City': 'Visakhapatnam', 'State': 'Andhra Pradesh', 'Region': 'South', 'Population_Density': 4000, 'Average_Income_Lakhs': 9, 'AQI_Severity': 52},
        {'City': 'Guwahati', 'State': 'Assam', 'Region': 'East', 'Population_Density': 3500, 'Average_Income_Lakhs': 7, 'AQI_Severity': 78},
        {'City': 'Nagpur', 'State': 'Maharashtra', 'Region': 'Central', 'Population_Density': 6000, 'Average_Income_Lakhs': 9, 'AQI_Severity': 69}
    ]
    city_df = pd.DataFrame(cities)

    # Calculate Risk Score
    # Normalizing factors to bring values into a comparable range for multiplication
    # Max values are used for simple min-max scaling (0-1)
    max_aqi = city_df['AQI_Severity'].max()
    max_pop_density = city_df['Population_Density'].max()
    max_income = city_df['Average_Income_Lakhs'].max()

    city_df['Normalized_AQI_Severity'] = city_df['AQI_Severity'] / max_aqi
    city_df['Normalized_Population_Density'] = city_df['Population_Density'] / max_pop_density
    city_df['Normalized_Average_Income'] = city_df['Average_Income_Lakhs'] / max_income

    city_df['Risk_Score'] = (
        city_df['Normalized_AQI_Severity'] *
        city_df['Normalized_Population_Density'] *
        city_df['Normalized_Average_Income']
    ) * 100 # Scale to 0-100 for easier interpretation

    # 2. Competitor Feature Data
    competitor_data = {
        'Brand': ['Philips', 'LG', 'Xiaomi', 'Dyson', 'Eureka Forbes'],
        'HEPA_Filter': [True, True, True, True, True],
        'Activated_Carbon': [True, True, True, True, True],
        'VOC_Sensor': [False, True, True, True, False],
        'PM25_Sensor': [True, True, True, True, True],
        'Smart_Control': [True, True, True, True, False],
        'App_Control': [True, True, True, True, False],
        'Compact_Design': [True, False, True, False, True],
        'Low_Noise': [True, True, True, False, True],
        'Affordable_Filters': [False, False, True, False, True],
        'Coverage_Area_SqFt': [400, 600, 450, 800, 350],
        'Price_Range_INR': ['15k-25k', '20k-35k', '8k-15k', '40k-70k', '10k-20k']
    }
    competitor_df = pd.DataFrame(competitor_data)

    return city_df, competitor_df

city_df, competitor_df = generate_simulated_data()

# --- Helper Functions for Download ---
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def create_prd_content(segment_features):
    prd_text = "## Product Requirements Document Insights\n\n"
    for segment, features in segment_features.items():
        prd_text += f"### {segment} Segment\n"
        prd_text += "#### Must-Have Features:\n"
        for feature, present in features.items():
            if present:
                prd_text += f"- [x] {feature}\n"
            else:
                prd_text += f"- [ ] {feature}\n"
        prd_text += "\n"
    return prd_text.encode('utf-8')

# --- Dashboard Layout ---
st.title("🌬️ Air Purifier Market Intelligence Dashboard")

st.sidebar.header("Navigation")
selected_tab = st.sidebar.radio(
    "Go to",
    ["Market Prioritization", "Health Cost Impact", "Competitor Analysis", "PRD Insights"]
)

# --- Module 1: Market Prioritization Dashboard ---
if selected_tab == "Market Prioritization":
    st.header("Market Prioritization Dashboard")
    st.write("Rank Indian Tier 1 & Tier 2 cities based on pollution impact using a calculated Risk Score.")

    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_regions = st.multiselect(
            "Select Region(s)",
            options=city_df['Region'].unique(),
            default=city_df['Region'].unique()
        )
    with col2:
        aqi_threshold = st.slider(
            "Minimum AQI Severity",
            min_value=int(city_df['AQI_Severity'].min()),
            max_value=int(city_df['AQI_Severity'].max()),
            value=int(city_df['AQI_Severity'].min())
        )
    with col3:
        pop_min, pop_max = int(city_df['Population_Density'].min()), int(city_df['Population_Density'].max())
        population_range = st.slider(
            "Population Density Range",
            min_value=pop_min,
            max_value=pop_max,
            value=(pop_min, pop_max)
        )

    filtered_cities_df = city_df[
        (city_df['Region'].isin(selected_regions)) &
        (city_df['AQI_Severity'] >= aqi_threshold) &
        (city_df['Population_Density'] >= population_range[0]) &
        (city_df['Population_Density'] <= population_range[1])
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    st.subheader("City Risk Scores")
    if not filtered_cities_df.empty:
        # Sortable Table
        st.dataframe(
            filtered_cities_df[['City', 'State', 'AQI_Severity', 'Population_Density', 'Average_Income_Lakhs', 'Risk_Score']]
            .sort_values(by='Risk_Score', ascending=False)
            .style.format({'Risk_Score': "{:.2f}"}),
            use_container_width=True
        )

        # Bar Chart of Top Cities by Risk Score
        fig_bar = px.bar(
            filtered_cities_df.sort_values(by='Risk_Score', ascending=False).head(10),
            x='City',
            y='Risk_Score',
            color='Risk_Score',
            color_continuous_scale=px.colors.sequential.Reds,
            title='Top Cities by Risk Score',
            labels={'Risk_Score': 'Risk Score (0-100)'},
            hover_data=['State', 'AQI_Severity', 'Population_Density', 'Average_Income_Lakhs']
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Interactive City-wise Bubble Map (using Plotly Go for more control)
        fig_map = go.Figure(data=go.Scattergeo(
            lon = filtered_cities_df['Average_Income_Lakhs'], # Using income as proxy for longitude for visual spread, not actual geo data
            lat = filtered_cities_df['Population_Density'], # Using pop density as proxy for latitude
            text = filtered_cities_df['City'] + '<br>Risk Score: ' + filtered_cities_df['Risk_Score'].round(2).astype(str),
            marker = dict(
                size = filtered_cities_df['Risk_Score'] / 2,  # Scale bubble size by risk score
                color = filtered_cities_df['AQI_Severity'], # Color by AQI Severity
                colorscale = 'Hot', # Choose a suitable colorscale
                line_color='rgb(40,40,40)',
                line_width=0.5,
                sizemode = 'area'
            )
        ))
        fig_map.update_layout(
            title_text = 'City Risk Score Bubble Map (Simulated Geo-Distribution)',
            showlegend = False,
            geo = dict(
                scope = 'asia', # Focus on Asia
                showland = True,
                landcolor = 'rgb(243,243,243)',
                countrycolor = 'rgb(204,204,204)',
            ),
            height=600 # Adjust height for better visibility
        )
        st.write("Note: The map uses simulated geographical distribution (income as pseudo-longitude, population density as pseudo-latitude) due to lack of actual city coordinates in dummy data. For real data, actual lat/lon would be used.")
        st.plotly_chart(fig_map, use_container_width=True)

        # Download option
        csv_data = convert_df_to_csv(filtered_cities_df)
        st.download_button(
            label="Download Filtered City Data as CSV",
            data=csv_data,
            file_name="filtered_city_risk_data.csv",
            mime="text/csv",
        )
    else:
        st.info("No cities match the selected filters. Please adjust your criteria.")

# --- Module 2: Health Cost Impact Projections ---
elif selected_tab == "Health Cost Impact":
    st.header("Health Cost Impact Projections")
    st.write("Estimate pollution-linked GDP loss per city, allocating the national economic burden of $36.8 Billion.")

    national_economic_burden_billion = 36.8 # Billion USD

    # Calculate weighted factor for each city based on AQI and Population
    city_df['Health_Impact_Factor'] = city_df['AQI_Severity'] * city_df['Population_Density']
    total_health_impact_factor = city_df['Health_Impact_Factor'].sum()

    # Allocate national burden proportionally
    city_df['Estimated_GDP_Loss_Million_USD'] = (
        (city_df['Health_Impact_Factor'] / total_health_impact_factor) * national_economic_burden_billion * 1000
    ).round(2)

    st.subheader("City-wise Estimated Economic Impact")

    # Bar chart for GDP loss
    fig_gdp_loss = px.bar(
        city_df.sort_values(by='Estimated_GDP_Loss_Million_USD', ascending=False).head(15),
        x='City',
        y='Estimated_GDP_Loss_Million_USD',
        color='Estimated_GDP_Loss_Million_USD',
        color_continuous_scale=px.colors.sequential.Plasma,
        title='Estimated Annual GDP Loss per City due to Air Pollution (Million USD)',
        labels={'Estimated_GDP_Loss_Million_USD': 'Estimated GDP Loss (Million USD)'},
        hover_data=['State', 'AQI_Severity', 'Population_Density']
    )
    st.plotly_chart(fig_gdp_loss, use_container_width=True)

    st.subheader("Comparison by State/Region")
    state_gdp_loss = city_df.groupby('State')['Estimated_GDP_Loss_Million_USD'].sum().reset_index()
    fig_state_gdp = px.bar(
        state_gdp_loss.sort_values(by='Estimated_GDP_Loss_Million_USD', ascending=False),
        x='State',
        y='Estimated_GDP_Loss_Million_USD',
        color='Estimated_GDP_Loss_Million_USD',
        color_continuous_scale=px.colors.sequential.Viridis,
        title='Estimated Annual GDP Loss by State (Million USD)',
        labels={'Estimated_GDP_Loss_Million_USD': 'Estimated GDP Loss (Million USD)'}
    )
    st.plotly_chart(fig_state_gdp, use_container_width=True)

    st.info(f"**Assumption:** National economic burden of $36.8 Billion (as per report) is proportionally distributed based on a city's combined AQI Severity and Population Density.")

    # Download option
    csv_data_gdp = convert_df_to_csv(city_df[['City', 'State', 'Estimated_GDP_Loss_Million_USD', 'AQI_Severity', 'Population_Density']])
    st.download_button(
        label="Download Estimated GDP Loss Data as CSV",
        data=csv_data_gdp,
        file_name="estimated_gdp_loss_data.csv",
        mime="text/csv",
    )

# --- Module 3: Competitor Feature Gap Matrix ---
elif selected_tab == "Competitor Analysis":
    st.header("Competitor Feature Gap Matrix")
    st.write("Analyze top 5 competitors across various features to identify market white space opportunities.")

    st.subheader("Feature Matrix Heatmap")

    # Prepare data for heatmap
    features_df = competitor_df.set_index('Brand')
    # Select only boolean/numeric feature columns for the heatmap
    feature_cols = [col for col in features_df.columns if features_df[col].dtype == 'bool' or pd.api.types.is_numeric_dtype(features_df[col])]

    # Convert boolean to int for heatmap color scale
    heatmap_data = features_df[feature_cols].replace({True: 1, False: 0})

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale=[[0, 'red'], [0.5, 'white'], [1, 'green']], # Red for False, Green for True
        text=[['✔' if val == 1 else '✖' for val in row] for row in heatmap_data.values],
        texttemplate="%{text}",
        hoverinfo='x+y+z'
    ))
    fig_heatmap.update_layout(
        title='Competitor Feature Presence',
        xaxis_title='Feature',
        yaxis_title='Brand',
        height=500
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.subheader("Textual Summary of Feature Gaps & White Space Opportunities")
    st.markdown("""
    Based on the simulated competitor data, here are some key observations and potential white space opportunities:

    * **VOC Sensors:** Only LG, Xiaomi, and Dyson currently offer VOC sensors. This is a significant gap for Philips and Eureka Forbes, representing a key opportunity for product differentiation, especially given the rising awareness of indoor air quality beyond just particulate matter.
    * **Smart Control & App Integration:** While most brands offer some form of smart control, Eureka Forbes lacks comprehensive app integration. This is a must-have for modern consumers seeking convenience and remote management.
    * **Compact Design with High Coverage:** Dyson, while offering high coverage, lacks a compact design. There's an opportunity for brands to develop high-CADR (Clean Air Delivery Rate) purifiers in more space-efficient form factors.
    * **Affordable Filters:** Dyson and Philips's filters are generally less affordable. Xiaomi and Eureka Forbes lead here. Offering cost-effective, long-lasting filter replacements can significantly improve customer loyalty and reduce the total cost of ownership, appealing to a broader market.
    * **Low Noise Operation:** Dyson is noted for potentially higher noise levels. Focusing on ultra-quiet operation, especially in sleep modes, remains a premium feature and a competitive advantage.

    **Overall White Space:**
    The market has room for products that combine **comprehensive VOC sensing, seamless smart app integration, a compact design suitable for smaller Indian homes, and genuinely affordable, long-lasting filters.** A brand that can deliver on all these fronts at a competitive price point (e.g., in the mid-range segment) could capture significant market share.
    """)

    # Download option
    csv_data_competitor = convert_df_to_csv(competitor_df)
    st.download_button(
        label="Download Competitor Feature Data as CSV",
        data=csv_data_competitor,
        file_name="competitor_feature_matrix.csv",
        mime="text/csv",
    )

# --- Module 4: Product Requirements Document (PRD) Insights ---
elif selected_tab == "PRD Insights":
    st.header("Product Requirements Document (PRD) Insights")
    st.write("Define air purifier requirements across different market segments (Mass, Mid-range, Premium).")

    # Define features for each segment
    segment_features = {
        "Mass Market (Sub-INR 15,000)": {
            "True HEPA (H13)": True,
            "Activated Carbon Filter": True,
            "PM2.5 Sensor": True,
            "VOC Sensor": False,
            "Smart Auto Mode": True,
            "App Control": False,
            "Scheduling": False,
            "Compact Design": True,
            "Low Noise (Sleep Mode)": True,
            "Affordable Filter Replacement": True,
            "Coverage (Sq. Ft.)": "200-350",
            "Target Income (Lakhs INR)": "<10"
        },
        "Mid-Range (INR 15,000 - 30,000)": {
            "True HEPA (H13+)": True,
            "Activated Carbon Filter": True,
            "PM2.5 Sensor": True,
            "VOC Sensor": True,
            "Smart Auto Mode": True,
            "App Control": True,
            "Scheduling": True,
            "Compact Design": True,
            "Low Noise (Sleep Mode)": True,
            "Affordable Filter Replacement": False, # Slightly less emphasis on 'affordable' compared to mass
            "Coverage (Sq. Ft.)": "350-600",
            "Target Income (Lakhs INR)": "10-15"
        },
        "Premium (Above INR 30,000)": {
            "True HEPA (H14)": True,
            "Advanced Activated Carbon": True,
            "PM2.5 Sensor": True,
            "VOC Sensor": True,
            "Smart Auto Mode": True,
            "App Control": True,
            "Scheduling": True,
            "Compact Design": False, # Can be larger for high CADR
            "Low Noise (Sleep Mode)": True,
            "Affordable Filter Replacement": False,
            "Coverage (Sq. Ft.)": "600+",
            "Target Income (Lakhs INR)": ">15"
        }
    }

    st.subheader("Interactive Feature Checklist per Segment")

    # Display features for each segment
    for segment, features in segment_features.items():
        st.markdown(f"### {segment}")
        col_features, col_income = st.columns([2, 1])
        with col_features:
            st.write("**Must-Have Features:**")
            for feature, present in features.items():
                if feature not in ["Coverage (Sq. Ft.)", "Target Income (Lakhs INR)"]:
                    st.checkbox(
    feature,
    value=present,
    disabled=True,
    help="This is a suggested feature for this segment.",
    key=f"{segment}_{feature}".replace(" ", "_")
)

        with col_income:
            st.write("**Target Market:**")
            st.markdown(f"**Coverage:** {features['Coverage (Sq. Ft.)']} sq. ft.")
            st.markdown(f"**Avg. Income:** {features['Target Income (Lakhs INR)']} Lakhs INR")
        st.markdown("---")

    st.subheader("Downloadable PRD Template")
    st.write("Download a basic PRD template outlining these features.")

    # Create a DataFrame for PRD download
    prd_df_data = []
    for segment, features in segment_features.items():
        row = {'Segment': segment}
        for feature, value in features.items():
            row[feature] = 'Yes' if value is True else ('No' if value is False else value)
        prd_df_data.append(row)
    prd_df = pd.DataFrame(prd_df_data)

    csv_prd = convert_df_to_csv(prd_df)
    st.download_button(
        label="Download PRD Template (CSV)",
        data=csv_prd,
        file_name="air_purifier_prd_template.csv",
        mime="text/csv",
    )

    # Option to download as Markdown (text file)
    markdown_prd = create_prd_content(segment_features)
    st.download_button(
        label="Download PRD Template (Markdown)",
        data=markdown_prd,
        file_name="air_purifier_prd_template.md",
        mime="text/markdown",
    )
    st.info("Note: PDF export is not directly supported by Streamlit without additional libraries. CSV and Markdown options are provided.")

# --- User Data Upload (Optional) ---
st.sidebar.markdown("---")
st.sidebar.header("Upload Your Data (Optional)")
uploaded_file = st.sidebar.file_uploader("Upload City Data (CSV)", type="csv")
if uploaded_file is not None:
    try:
        user_city_df = pd.read_csv(uploaded_file)
        # Basic validation and recalculation of risk score for uploaded data
        required_cols = ['City', 'State', 'Region', 'AQI_Severity', 'Population_Density', 'Average_Income_Lakhs']
        if all(col in user_city_df.columns for col in required_cols):
            st.sidebar.success("Custom city data uploaded successfully!")
            # Recalculate risk score for user data
            max_aqi_user = user_city_df['AQI_Severity'].max()
            max_pop_density_user = user_city_df['Population_Density'].max()
            max_income_user = user_city_df['Average_Income_Lakhs'].max()

            user_city_df['Normalized_AQI_Severity'] = user_city_df['AQI_Severity'] / max_aqi_user
            user_city_df['Normalized_Population_Density'] = user_city_df['Population_Density'] / max_pop_density_user
            user_city_df['Normalized_Average_Income'] = user_city_df['Average_Income_Lakhs'] / max_income_user
            user_city_df['Risk_Score'] = (
                user_city_df['Normalized_AQI_Severity'] *
                user_city_df['Normalized_Population_Density'] *
                user_city_df['Normalized_Average_Income']
            ) * 100
            st.session_state['city_df'] = user_city_df # Store in session state to use across modules
            st.sidebar.info("Dashboard now using your uploaded data.")
        else:
            st.sidebar.error(f"Uploaded CSV must contain columns: {', '.join(required_cols)}")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")

# Initialize session state for city_df if not already set by upload
if 'city_df' not in st.session_state:
    st.session_state['city_df'] = generate_simulated_data()[0]

# Use the city_df from session_state for all modules
city_df = st.session_state['city_df']