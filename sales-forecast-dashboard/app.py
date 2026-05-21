import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import math

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSmodels_AVAILABLE = True
except Exception:
    STATSmodels_AVAILABLE = False


PRIMARY = "#3B82F6"
BACKGROUND = "#0F172A"
CARD = "#1E293B"
TEXT = "#F8FAFC"
ACCENT_GREEN = "#10B981"
ACCENT_ORANGE = "#F97316"
MUTED_BLUE = "#64748B"

DATA_PATH = "sales-forecast-dashboard/data/superstore.csv"


def get_template():
    """Create a professional Plotly template with custom colors"""
    return {
        'layout': {
            'template': 'plotly_dark',
            'paper_bgcolor': CARD,
            'plot_bgcolor': CARD,
            'font': {'family': 'Arial, sans-serif', 'size': 12, 'color': TEXT},
            'title': {'font': {'size': 16, 'color': TEXT}},
            'xaxis': {'showgrid': True, 'gridwidth': 1, 'gridcolor': 'rgba(100, 116, 139, 0.2)'},
            'yaxis': {'showgrid': True, 'gridwidth': 1, 'gridcolor': 'rgba(100, 116, 139, 0.2)'},
        }
    }


def apply_professional_styling(fig):
    fig.update_layout(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(family='Arial, sans-serif', size=12, color=TEXT),
        title_font=dict(size=16, color=TEXT),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100, 116, 139, 0.2)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100, 116, 139, 0.2)'),
        hovermode='x unified',
        margin=dict(l=50, r=50, t=50, b=50),
    )
    return fig


@st.cache_data
def load_data(path=DATA_PATH):
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        st.error(f"Failed loading data: {e}\nPlease place your dataset at {path}")
        return None
    df.columns = [c.strip() for c in df.columns]

    for col in ("Order Date", "Ship Date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "Order Date" in df.columns:
        df["Order Year"] = df["Order Date"].dt.year
        df["Order Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()

    for c in ["Sales", "Profit", "Quantity"]:
        if c not in df.columns:
            df[c] = 0
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df


def safe_sum(df, col):
    return float(df[col].sum()) if col in df.columns else 0.0


def render_kpi_card(title, value, delta=None, delta_label=None, col=None, icon=""):
    """Render a professional KPI card with custom styling"""
    metric_container = col if col is not None else st
    
    delta_text = ""
    delta_color = "off"
    if delta is not None and not math.isnan(delta):
        delta_text = f"{delta:+.1%}"
        delta_color = "inverse" if delta < 0 else "off"
    
    metric_container.metric(
        label=f"{icon} {title}",
        value=value,
        delta=delta_text if delta_text else None,
        delta_color=delta_color
    )


def kpi_cards(df):
    """Professional KPI cards with better layout and styling"""
    revenue = safe_sum(df, "Sales")
    orders = df["Order ID"].nunique() if "Order ID" in df.columns else len(df)
    aov = revenue / orders if orders else 0
    profit = safe_sum(df, "Profit")

    yoy = None
    if "Order Year" in df.columns:
        years = df.groupby("Order Year")["Sales"].sum().sort_index()
        if len(years) >= 2:
            yoy = (years.iloc[-1] - years.iloc[-2]) / years.iloc[-2] if years.iloc[-2] != 0 else math.nan

    col1, col2, col3, col4 = st.columns(4, gap="medium")
    
    render_kpi_card("Total Revenue", f"${revenue:,.0f}", None, None, col1, "")
    render_kpi_card("Total Orders", f"{orders:,}", None, None, col2, "")
    render_kpi_card("Avg Order Value", f"${aov:,.2f}", None, None, col3, "")
    render_kpi_card("Profit Margin", f"{(profit/revenue*100) if revenue else 0:.1f}%", yoy, "YoY", col4, "")


def timeseries_chart(df, measure="Sales", smooth=True):
    """Professional time series chart"""
    if "Order Month" not in df.columns:
        st.info("No order date available for time series")
        return
    ts = df.groupby("Order Month")[measure].sum().reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts["Order Month"],
        y=ts[measure],
        mode='lines',
        name=measure,
        line=dict(color=PRIMARY, width=3, shape='spline' if smooth else 'linear'),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)',
        hovertemplate='<b>%{x|%b %Y}</b><br>' + measure + ': $%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"<b>{measure} Trend</b>",
        xaxis_title="Date",
        yaxis_title=measure,
        height=400,
        hovermode='x unified',
        showlegend=False
    )
    fig = apply_professional_styling(fig)
    st.plotly_chart(fig, use_container_width=True)


def bar_by_region(df, measure="Sales"):
    """Professional horizontal bar chart"""
    if "Region" not in df.columns:
        st.info("No Region column available")
        return
    reg = df.groupby("Region")[measure].sum().reset_index().sort_values(measure)
    
    fig = go.Figure(data=[
        go.Bar(
            y=reg["Region"],
            x=reg[measure],
            orientation='h',
            marker=dict(
                color=reg[measure],
                colorscale=[[0, PRIMARY], [1, ACCENT_GREEN]],
                showscale=False
            ),
            text=reg[measure].apply(lambda x: f'${x:,.0f}'),
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>' + measure + ': $%{x:,.0f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=f"<b>{measure} by Region</b>",
        xaxis_title=measure,
        yaxis_title="Region",
        height=400,
        showlegend=False
    )
    fig = apply_professional_styling(fig)
    st.plotly_chart(fig, use_container_width=True)


def category_performance(df):
    """Professional category sales chart"""
    if "Category" not in df.columns:
        st.info("No Category column available")
        return
    cat = df.groupby("Category")["Sales"].sum().reset_index().sort_values("Sales")
    
    colors = [ACCENT_GREEN, ACCENT_ORANGE, PRIMARY]
    
    fig = go.Figure(data=[
        go.Bar(
            y=cat["Category"],
            x=cat["Sales"],
            orientation='h',
            marker=dict(color=colors[:len(cat)]),
            text=cat["Sales"].apply(lambda x: f'${x:,.0f}'),
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Sales: $%{x:,.0f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="<b>Sales by Category</b>",
        xaxis_title="Sales",
        yaxis_title="Category",
        height=350,
        showlegend=False
    )
    fig = apply_professional_styling(fig)
    st.plotly_chart(fig, use_container_width=True)


def top_products(df, top_n=10):
    """Professional top products chart"""
    if "Product Name" not in df.columns:
        st.info("No Product Name column available")
        return
    top = df.groupby("Product Name")["Sales"].sum().nlargest(top_n).reset_index()
    
    fig = go.Figure(data=[
        go.Bar(
            y=top["Product Name"],
            x=top["Sales"],
            orientation='h',
            marker=dict(
                color=PRIMARY,
                line=dict(color=TEXT, width=1)
            ),
            text=top["Sales"].apply(lambda x: f'${x:,.0f}'),
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Sales: $%{x:,.0f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=f"<b>Top {top_n} Products</b>",
        xaxis_title="Sales",
        yaxis_title="Product",
        height=500,
        showlegend=False
    )
    fig = apply_professional_styling(fig)
    st.plotly_chart(fig, use_container_width=True)


def shipping_mode_distribution(df):
    """Professional shipping mode donut chart"""
    if "Ship Mode" not in df.columns:
        st.info("No Ship Mode column available")
        return
    sm = df["Ship Mode"].value_counts().reset_index()
    sm.columns = ["Ship Mode", "Count"]
    
    colors = [PRIMARY, ACCENT_GREEN, ACCENT_ORANGE, MUTED_BLUE]
    
    fig = go.Figure(data=[go.Pie(
        labels=sm["Ship Mode"],
        values=sm["Count"],
        hole=0.5,
        marker=dict(colors=colors[:len(sm)], line=dict(color=CARD, width=3)),
        textposition='inside',
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title="<b>Shipping Mode Distribution</b>",
        height=400,
        showlegend=True
    )
    fig = apply_professional_styling(fig)
    st.plotly_chart(fig, use_container_width=True)


def customer_segments(df):
    """Professional customer segments chart"""
    if "Segment" not in df.columns:
        st.info("No Segment column available")
        return
    seg = df.groupby("Segment")["Sales"].sum().reset_index().sort_values("Sales")
    
    colors = [PRIMARY, ACCENT_GREEN, ACCENT_ORANGE]
    
    fig = go.Figure(data=[
        go.Bar(
            y=seg["Segment"],
            x=seg["Sales"],
            orientation='h',
            marker=dict(color=colors[:len(seg)]),
            text=seg["Sales"].apply(lambda x: f'${x:,.0f}'),
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Sales: $%{x:,.0f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="<b>Sales by Customer Segment</b>",
        xaxis_title="Sales",
        yaxis_title="Segment",
        height=350,
        showlegend=False
    )
    fig = apply_professional_styling(fig)
    st.plotly_chart(fig, use_container_width=True)


def forecast_sales(df, periods=30, measure="Sales"):
    """Generate professional sales forecast"""
    # Use monthly totals
    if "Order Month" not in df.columns:
        st.info("Order date required for forecasting")
        return None
    ts = df.groupby("Order Month")[measure].sum().asfreq('MS')
    ts = ts.fillna(0)

    if STATSmodels_AVAILABLE:
        try:
            model = ExponentialSmoothing(ts, trend='add', seasonal=None)
            fit = model.fit(optimized=True)
            pred = fit.forecast(periods)
            res = fit.resid
            sigma = np.nanstd(res)
            index = pd.date_range(ts.index[-1] + pd.offsets.MonthBegin(1), periods=periods, freq='MS')
            forecast_df = pd.DataFrame({ 
                'ds': index, 
                'yhat': pred.values, 
                'yhat_lower': pred.values - 1.96*sigma, 
                'yhat_upper': pred.values + 1.96*sigma 
            })
            return forecast_df, ts
        except Exception:
            pass

    roll = ts.rolling(3).mean().dropna()
    level = roll.iloc[-1] if len(roll) else ts.mean()
    index = pd.date_range(ts.index[-1] + pd.offsets.MonthBegin(1), periods=periods, freq='MS')
    preds = np.repeat(level, periods)
    resid = ts - ts.rolling(3).mean()
    sigma = np.nanstd(resid.dropna()) if len(resid.dropna())>0 else 0
    forecast_df = pd.DataFrame({ 
        'ds': index, 
        'yhat': preds, 
        'yhat_lower': preds - 1.96*sigma, 
        'yhat_upper': preds + 1.96*sigma 
    })
    return forecast_df, ts


def render_forecast_chart(forecast_df, hist):
    """Render professional forecast chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist.values,
        name='Historical',
        mode='lines',
        line=dict(color=PRIMARY, width=3),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)',
        hovertemplate='<b>%{x|%b %Y}</b><br>Sales: $%{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat'],
        name='Forecast',
        mode='lines',
        line=dict(color=ACCENT_GREEN, width=3, dash='dash'),
        hovertemplate='<b>%{x|%b %Y}</b><br>Forecast: $%{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_upper'],
        name='Upper Bound',
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hovertemplate='<b>%{x|%b %Y}</b><br>Upper: $%{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_lower'],
        name='95% Confidence Interval',
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(16, 185, 129, 0.1)',
        fill='tonexty',
        hovertemplate='<b>%{x|%b %Y}</b><br>Lower: $%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="<b>Sales Forecast with Confidence Interval</b>",
        xaxis_title="Date",
        yaxis_title="Sales",
        height=450,
        hovermode='x unified'
    )
    fig = apply_professional_styling(fig)
    st.plotly_chart(fig, use_container_width=True)


US_STATE_ABBREV = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO','Connecticut':'CT','Delaware':'DE','District of Columbia':'DC','Florida':'FL','Georgia':'GA','Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI','Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND','Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA','Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'
}


def choropleth_by_state(df, measure="Sales"):
    """Professional choropleth map"""
    if "State" not in df.columns:
        st.info("No State column available for geographic insights")
        return
    s = df.groupby("State")[measure].sum().reset_index()
    s['state_abbrev'] = s['State'].map(US_STATE_ABBREV)
    s = s.dropna(subset=['state_abbrev'])
    if s.empty:
        st.info("No mappable states found")
        return
    
    fig = px.choropleth(
        s, 
        locations='state_abbrev', 
        locationmode='USA-states', 
        color=measure, 
        scope='usa',
        title=f'<b>{measure} by State</b>',
        color_continuous_scale=[[0, CARD], [1, PRIMARY]],
        hover_data={measure: ':,.0f', 'state_abbrev': False, 'State': True}
    )
    
    fig.update_layout(height=500)
    fig = apply_professional_styling(fig)
    st.plotly_chart(fig, use_container_width=True)


def customer_analytics(df):
    """Professional customer analytics section"""
    if "Customer ID" not in df.columns and "Customer Name" not in df.columns:
        st.info("No customer identifiers available")
        return
    cid = 'Customer ID' if 'Customer ID' in df.columns else 'Customer Name'
    cust = df.groupby(cid).agg(
        orders=('Order ID','nunique') if 'Order ID' in df.columns else ('Sales','count'), 
        revenue=('Sales','sum')
    ).reset_index()
    repeat_rate = (cust['orders']>1).mean()
    
    col1, col2, col3 = st.columns(3)
    render_kpi_card("Repeat Customer Rate", f"{repeat_rate:.1%}", None, None, col1, "")
    render_kpi_card("Avg Customer Value", f"${cust['revenue'].mean():,.0f}", None, None, col2, "")
    render_kpi_card("Total Customers", f"{len(cust):,}", None, None, col3, "")

    st.markdown("---")
    
    clv = cust.sort_values('revenue', ascending=False).head(10)
    fig = go.Figure(data=[
        go.Bar(
            y=clv[cid].astype(str),
            x=clv['revenue'],
            orientation='h',
            marker=dict(
                color=clv['revenue'],
                colorscale=[[0, PRIMARY], [1, ACCENT_GREEN]],
                showscale=True,
                colorbar=dict(title="Revenue", thickness=15)
            ),
            text=clv['revenue'].apply(lambda x: f'${x:,.0f}'),
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Lifetime Value: $%{x:,.0f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="<b>Top 10 Customers by Lifetime Value</b>",
        xaxis_title="Revenue",
        yaxis_title="Customer",
        height=400
    )
    fig = apply_professional_styling(fig)
    st.plotly_chart(fig, use_container_width=True)

    if 'Order Date' in df.columns and 'Ship Date' in df.columns:
        df_copy = df.copy()
        df_copy['processing_days'] = (pd.to_datetime(df_copy['Ship Date']) - pd.to_datetime(df_copy['Order Date'])).dt.days
        proc = df_copy['processing_days'].dropna()
        if not proc.empty:
            fig2 = go.Figure(data=[
                go.Histogram(
                    x=proc,
                    nbinsx=30,
                    marker=dict(color=PRIMARY, line=dict(color=TEXT, width=1)),
                    hovertemplate='Processing time: %{x} days<br>Count: %{y}<extra></extra>'
                )
            ])
            fig2.update_layout(
                title="<b>Order Processing Time Distribution</b>",
                xaxis_title="Days",
                yaxis_title="Frequency",
                height=350,
                showlegend=False
            )
            fig2 = apply_professional_styling(fig2)
            st.plotly_chart(fig2, use_container_width=True)


def main():
    st.set_page_config(
        page_title="Sales Analytics Dashboard",
        layout='wide',
        initial_sidebar_state='expanded'
    )
    
    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-color: {BACKGROUND};
    }}
    
    [data-testid="stSidebar"] {{
        background-color: {CARD};
    }}
    
    .main {{
        color: {TEXT};
        background-color: {BACKGROUND};
    }}
    
    .stMetricValue {{
        font-size: 32px;
        font-weight: 700;
        color: {PRIMARY};
    }}
    
    .stMetricLabel {{
        font-size: 14px;
        font-weight: 600;
        color: {MUTED_BLUE};
    }}
    
    .stDivider {{
        border-color: {CARD};
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: {TEXT};
        font-weight: 700;
    }}
    
    .stSelectbox label, .stDateInput label, .stMultiSelect label {{
        color: {TEXT};
        font-weight: 600;
        font-size: 12px;
    }}
    
    [data-baseweb="select"]  {{
        background-color: {CARD};
    }}
    
    [data-baseweb="input"] {{
        background-color: {CARD};
        color: {TEXT};
    }}
    
    [data-baseweb="tab"]  {{
        background-color: transparent;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.markdown(f"<h1 style='margin-bottom: 0;'>Sales Analytics Dashboard</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {MUTED_BLUE}; margin-top: -10px;'>Real-time business intelligence & forecasting</p>", unsafe_allow_html=True)
    
    df = load_data()
    if df is None:
        return

    st.sidebar.markdown("<h2 style='text-align: center; color: " + PRIMARY + ";'>NAVIGATION</h2>", unsafe_allow_html=True)
    page = st.sidebar.radio(
        "Select View",
        ["Executive Overview", "Forecasting", "Geographic Insights", "Customer Analytics"],
        label_visibility="collapsed",
        horizontal=False
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h3 style='color: " + TEXT + ";'>FILTERS</h3>", unsafe_allow_html=True)
    
    date_range = None
    if 'Order Date' in df.columns:
        min_date = df['Order Date'].min().date()
        max_date = df['Order Date'].max().date()
        date_range = st.sidebar.date_input(
            'Date Range',
            [min_date, max_date],
            format="MM/DD/YYYY"
        )
    
    regions = df['Region'].dropna().unique().tolist() if 'Region' in df.columns else []
    region = st.sidebar.multiselect(
        'Region',
        options=regions,
        default=regions,
        label_visibility="visible"
    )

    categories = df['Category'].dropna().unique().tolist() if 'Category' in df.columns else []
    category = st.sidebar.multiselect(
        'Category',
        options=categories,
        default=categories,
        label_visibility="visible"
    )

    filtered = df.copy()
    if date_range and len(date_range) == 2 and 'Order Date' in df.columns:
        start, end = date_range
        filtered = filtered[(filtered['Order Date'] >= pd.to_datetime(start)) & (filtered['Order Date'] <= pd.to_datetime(end))]
    if regions:
        filtered = filtered[filtered['Region'].isin(region)]
    if categories:
        filtered = filtered[filtered['Category'].isin(category)]

    if page == 'Executive Overview':
        st.markdown("<h2>Executive Summary</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {MUTED_BLUE}; margin-top: -10px;'>Key performance indicators and business metrics</p>", unsafe_allow_html=True)
        
        kpi_cards(filtered)
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1.2], gap="large")
        with col1:
            timeseries_chart(filtered, measure='Sales', smooth=True)
        with col2:
            bar_by_region(filtered)

        st.markdown("---")

        col3, col4 = st.columns(2, gap="large")
        with col3:
            category_performance(filtered)
        with col4:
            top_products(filtered, top_n=10)

        st.markdown("---")
        
        col5, col6 = st.columns(2, gap="large")
        with col5:
            shipping_mode_distribution(filtered)
        with col6:
            customer_segments(filtered)

    elif page == 'Forecasting':
        st.markdown("<h2>Sales Forecasting</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {MUTED_BLUE}; margin-top: -10px;'>AI-powered predictions with confidence intervals</p>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            horizon = st.selectbox(
                'Forecast Horizon',
                [30, 60, 90, 120],
                index=0,
                help="Number of months to forecast"
            )
        with col2:
            st.info(f"Forecasting **{horizon} months** ahead")
        
        st.markdown("---")
        
        with st.spinner('Computing forecast...'):
            result = forecast_sales(filtered, periods=horizon, measure='Sales')
        
        if result:
            forecast_df, hist = result
            render_forecast_chart(forecast_df, hist)
            
            # Forecast statistics
            st.markdown("---")
            st.markdown("<h3>Forecast Summary</h3>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            avg_forecast = forecast_df['yhat'].mean()
            max_forecast = forecast_df['yhat'].max()
            min_forecast = forecast_df['yhat'].min()
            growth = (forecast_df['yhat'].iloc[-1] - hist.iloc[-1]) / hist.iloc[-1] * 100
            
            render_kpi_card("Avg Forecast", f"${avg_forecast:,.0f}", None, None, col1, "")
            render_kpi_card("Peak Month", f"${max_forecast:,.0f}", None, None, col2, "")
            render_kpi_card("Low Month", f"${min_forecast:,.0f}", None, None, col3, "")
            render_kpi_card("Expected Growth", f"{growth:+.1f}%", None, None, col4, "")
        else:
            st.warning('Forecasting could not be performed with the current data or installed libraries.')

    elif page == 'Geographic Insights':
        st.markdown("<h2>Geographic Analysis</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {MUTED_BLUE}; margin-top: -10px;'>Regional performance and market distribution</p>", unsafe_allow_html=True)
        
        choropleth_by_state(filtered)
        st.markdown("---")
        
        st.markdown("<h3>Top Performing States</h3>", unsafe_allow_html=True)
        if 'State' in filtered.columns:
            top_states = filtered.groupby('State')['Sales'].sum().nlargest(10).reset_index()
            
            fig = go.Figure(data=[
                go.Bar(
                    x=top_states['State'],
                    y=top_states['Sales'],
                    marker=dict(
                        color=top_states['Sales'],
                        colorscale=[[0, PRIMARY], [1, ACCENT_GREEN]],
                        showscale=False
                    ),
                    text=top_states['Sales'].apply(lambda x: f'${x:,.0f}'),
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Sales: $%{y:,.0f}<extra></extra>'
                )
            ])
            fig.update_layout(
                xaxis_title="State",
                yaxis_title="Sales",
                height=400,
                showlegend=False
            )
            fig = apply_professional_styling(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No state-level data available')

    elif page == 'Customer Analytics':
        st.markdown("<h2>Customer Intelligence</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {MUTED_BLUE}; margin-top: -10px;'>Customer behavior, lifetime value, and retention metrics</p>", unsafe_allow_html=True)
        
        customer_analytics(filtered)

    st.markdown("---")
    footer_col1, footer_col2 = st.columns([0.7, 0.3])
    with footer_col1:
        st.markdown(f"<p style='color: {MUTED_BLUE}; font-size: 12px;'><b>Data Preview</b> • Showing sample of filtered dataset</p>", unsafe_allow_html=True)
        st.dataframe(
            filtered.sample(min(len(filtered), 500)),
            use_container_width=True,
            height=300
        )
    
    with footer_col2:
        csv = filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='Download Data',
            data=csv,
            file_name='sales_data.csv',
            mime='text/csv',
            use_container_width=True
        )
        
        st.markdown(f"""
        <p style='color: {MUTED_BLUE}; font-size: 11px; margin-top: 20px;'>
        <b>Dashboard Stats</b><br>
        Records: {len(filtered):,} / {len(df):,}<br>
        Date Range: {filtered['Order Date'].min().strftime('%b %Y') if 'Order Date' in filtered.columns else 'N/A'}
        </p>
        """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
