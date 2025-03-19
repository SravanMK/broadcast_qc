import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards
from datetime import datetime

# Define status colors
STATUS_COLORS = {
    'Complete': '#28A745',  # Vibrant Green
    'No Game': '#FF5733',  # Rich Red-Orange
    'Incomplete': '#FFC107',  # Warm Yellow
    'Wrong Game': '#FF6F61',  # Stylish Coral Red
    'In Progress': '#007BFF'  # Bright Blue
}

# Custom CSS for metric cards
CARD_STYLE = '''
<style>
    .metric-card {
        background-color: #333333;  /* Dark Background */
        color: #f0f0f0;  /* Light Text Color */
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 15px;
    }
</style>
'''


# Load data function
def load_data():
    file_path = r"C:\Users\user\PycharmProjects\PythonProject\broadcast_qc_dashboard\Broadcast QC - Sheet30.csv"
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df


# Main function
def main():
    st.set_page_config(page_title='Team Performance Dashboard', layout='wide')

    # Apply custom CSS
    st.markdown(CARD_STYLE, unsafe_allow_html=True)
    st.markdown('<div class="main-title">Team Performance Dashboard</div>', unsafe_allow_html=True)

    # Load data
    df = load_data()

    # Sidebar filters
    st.sidebar.header('Filters')
    home_teams = sorted(df['Home Team'].dropna().unique().tolist())
    away_teams = sorted(df['Away Team'].dropna().unique().tolist())
    channels = sorted(df['Channel'].dropna().unique().tolist())
    networks = sorted(df['Network'].dropna().unique().tolist())

    selected_home_team = st.sidebar.selectbox('Select Home Team', ['All'] + home_teams)
    selected_away_team = st.sidebar.selectbox('Select Away Team', ['All'] + away_teams)
    selected_channel = st.sidebar.selectbox('Select Channel', ['All'] + channels)
    selected_network = st.sidebar.selectbox('Select Network', ['All'] + networks)

    # Date range filter
    min_date = df['Date'].min().date() if not df['Date'].isna().all() else None
    max_date = df['Date'].max().date() if not df['Date'].isna().all() else None
    date_option = st.sidebar.radio('Date Range', ['All', 'Custom'], index=0)

    if date_option == 'All' and min_date and max_date:
        start_date, end_date = min_date, max_date
    elif date_option == 'Custom':
        if min_date and max_date:
            start_date, end_date = st.sidebar.date_input('Select Date Range', [min_date, max_date], key='date_range')
        else:
            st.sidebar.warning("No valid date range found in the data.")
            start_date, end_date = None, None

    # Filter data based on selections
    filtered_df = df.copy()
    if selected_home_team != 'All':
        filtered_df = filtered_df[filtered_df['Home Team'] == selected_home_team]
    if selected_away_team != 'All':
        filtered_df = filtered_df[filtered_df['Away Team'] == selected_away_team]
    if selected_channel != 'All':
        filtered_df = filtered_df[filtered_df['Channel'] == selected_channel]
    if selected_network != 'All':
        filtered_df = filtered_df[filtered_df['Network'] == selected_network]
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['Date'].dt.date >= start_date) & (filtered_df['Date'].dt.date <= end_date)]

    # Calculate metrics
    total_records = len(filtered_df)
    complete_records = filtered_df[filtered_df['Status'] == 'Complete'].shape[0]
    complete_percentage = (complete_records / total_records * 100) if total_records > 0 else 0

    # Display metrics
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><h3>Total Recordings</h3><h1>{total_records}</h1></div>',
                        unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h3>Total Complete</h3><h1>{complete_records}</h1></div>',
                        unsafe_allow_html=True)
        with col3:
            st.markdown(
                f'<div class="metric-card"><h3>Complete Percentage</h3><h1>{complete_percentage:.2f}%</h1></div>',
                unsafe_allow_html=True)
    style_metric_cards()

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(['📊 Status Overview', '📈 Performance Analysis', '📅 Daily Trends'])

    # Tab 1: Status Overview
    with tab1:
        st.subheader('Filtered Data')
        status_counts = filtered_df['Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']

        col1, col2 = st.columns([1.5, 3])
        with col1:
            fig = px.pie(status_counts, values='Count', names='Status', title='Status Distribution',
                         color='Status', color_discrete_map=STATUS_COLORS, width=480, height=480, hole=0.3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig)

        with col2:
            complete_over_time = filtered_df.groupby('Date')['Status'].apply(
                lambda x: (x == 'Complete').sum() / len(x) * 100).reset_index(name='Complete %')
            fig_line = px.line(complete_over_time, x='Date', y='Complete %', title='Completion % Over Time', width=1100)
            st.plotly_chart(fig_line)

        # Download button for filtered data
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="filtered_data.csv",
            mime="text/csv",
        )

        # Display the filtered data table
        st.subheader('Filtered Data Table')
        st.dataframe(filtered_df)

    # Tab 2: Performance Analysis
    with tab2:
        st.subheader('Channel-wise Performance')

        # Calculate total records per channel
        total_per_channel = filtered_df.groupby('Channel').size().reset_index(name='Total_Channel')

        # Calculate complete records per channel and iptv combination
        complete_per_channel_iptv = filtered_df[filtered_df['Status'] == 'Complete'] \
            .groupby(['Channel', 'iptv']).size().reset_index(name='Complete_Count')

        # Merge totals and complete counts
        merged = pd.merge(complete_per_channel_iptv, total_per_channel, on='Channel', how='right')

        # Calculate contribution percentage (fill NaN with 0 for channels with no completions)
        merged['Contribution %'] = (merged['Complete_Count'].fillna(0) / merged['Total_Channel']) * 100

        # Calculate total completion % per channel for sorting
        total_completion = merged.groupby('Channel')['Contribution %'].sum().reset_index(name='Total_Completion %')

        # Merge back for sorting
        final_df = pd.merge(merged, total_completion, on='Channel').sort_values('Total_Completion %', ascending=False)

        # Create the sorted stacked bar chart
        fig = px.bar(final_df,
                     x='Channel',
                     y='Contribution %',
                     color='iptv',
                     title='Channel Performance: Completion % Contribution by IPTV',
                     labels={'Contribution %': 'Contribution to Total Completion (%)'},
                     hover_data=['Total_Channel', 'Complete_Count'],
                     barmode='stack',
                     color_discrete_sequence=px.colors.qualitative.Pastel)

        # Add total completion % as text above bars
        fig.add_trace(go.Scatter(
            x=total_completion['Channel'],
            y=total_completion['Total_Completion %'] + 7,  # Offset for visibility
            text=total_completion['Total_Completion %'].round(2).astype(str) + '%',
            mode='text',
            showlegend=False
        ))

        st.plotly_chart(fig)

    # Tab 3: Daily Trends
    with tab3:
        st.subheader('Daily Trends of Statuses')

        # Group by Date, Status, Home Team, and Away Team
        daily_status_counts = filtered_df.groupby(['Date', 'Status', 'Home Team', 'Away Team']).size().reset_index(
            name='Count')

        # Convert pandas.Timestamp to datetime.date for the slider
        min_date_tab3 = daily_status_counts['Date'].min().date()  # Convert to datetime.date
        max_date_tab3 = daily_status_counts['Date'].max().date()  # Convert to datetime.date

        # Check if min_date and max_date are the same
        if min_date_tab3 == max_date_tab3:
            st.warning("Only one date of data is available. The slider is disabled.")
            selected_dates = (min_date_tab3, max_date_tab3)  # Use the single date directly
        else:
            # Add a date slider for filtering
            selected_dates = st.slider(
                'Select Date Range',
                min_value=min_date_tab3,
                max_value=max_date_tab3,
                value=(min_date_tab3, max_date_tab3)
            )

        # Filter data based on the selected date range
        filtered_daily_status_counts = daily_status_counts[
            (daily_status_counts['Date'].dt.date >= selected_dates[0]) &
            (daily_status_counts['Date'].dt.date <= selected_dates[1])
            ]

        # Plot the chart
        fig2 = px.bar(filtered_daily_status_counts,
                      x='Date',
                      y='Count',
                      color='Status',
                      title='Daily Trends of Statuses',
                      color_discrete_map=STATUS_COLORS,
                      barmode='stack',
                      hover_data=['Home Team', 'Away Team'])  # Add hover data
        st.plotly_chart(fig2)


# Run the app
if __name__ == '__main__':
    main()