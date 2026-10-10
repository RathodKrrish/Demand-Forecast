import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import xgboost as xgb

warnings.filterwarnings("ignore", category=RuntimeWarning)
st.set_page_config(page_title="Store Demand Forecast", page_icon="📈", layout="wide")
st.title(" Store Demand Forecast")
st.caption("Explore historical store-item sales, train XGBoost, and export demand predictions.")

FEATURES = ["day","month","year","dayofweek","dayofyear","weekofyear",
            "sales_mean","sales_median","sales_std","lag_7","lag_14","lag_28","lag_365"]

def smape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    den = np.abs(y) + np.abs(p)
    e = np.divide(2*np.abs(y-p), den, out=np.zeros_like(den), where=den != 0)
    return float(100*e.mean())

def calendar(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["day"] = df.date.dt.day
    df["month"] = df.date.dt.month
    df["year"] = df.date.dt.year
    df["dayofweek"] = df.date.dt.dayofweek
    df["dayofyear"] = df.date.dt.dayofyear
    df["weekofyear"] = df.date.dt.isocalendar().week.astype("int16")
    return df

def prepare(train, test):
    train, test = calendar(train), calendar(test)
    stats = train.groupby(["store","item"]).sales.agg(
        sales_mean="mean", sales_median="median", sales_std="std").reset_index()
    stats["sales_std"] = stats.sales_std.fillna(0)
    train = train.merge(stats, on=["store","item"], how="left")
    test = test.merge(stats, on=["store","item"], how="left")
    train = train.sort_values(["store","item","date"]).reset_index(drop=True)
    test = test.sort_values(["date","store","item"]).reset_index(drop=True)
    for lag in [7,14,28,365]:
        train[f"lag_{lag}"] = train.groupby(["store","item"]).sales.shift(lag)
    return train, test

def forecast_sales(train, test, rounds, eta, depth):
    tr, te = prepare(train, test)
    cutoff = tr.date.max() - pd.Timedelta(days=90)
    data = tr.dropna(subset=["lag_7"]).copy()
    fit = data[data.date < cutoff]
    val = data[data.date >= cutoff]
    if fit.empty or val.empty:
        raise ValueError("Need enough history for a 90-day validation split.")
    params = {"objective":"reg:squarederror","eval_metric":"rmse","eta":eta,
              "max_depth":depth,"subsample":0.9,"colsample_bytree":0.9,
              "seed":42,"nthread":2}
    dfit = xgb.DMatrix(fit[FEATURES].fillna(0), label=np.log1p(fit.sales), feature_names=FEATURES)
    dval = xgb.DMatrix(val[FEATURES].fillna(0), label=np.log1p(val.sales), feature_names=FEATURES)
    model = xgb.train(params, dfit, num_boost_round=rounds, verbose_eval=False)
    val_pred = np.clip(np.expm1(model.predict(dval)), 0, None)
    model_score = smape(val.sales, val_pred)
    baseline_score = smape(val.sales, np.full(len(val), fit.sales.mean()))

    full = tr.dropna(subset=["lag_7"])
    dfull = xgb.DMatrix(full[FEATURES].fillna(0), label=np.log1p(full.sales), feature_names=FEATURES)
    final_model = xgb.train(params, dfull, num_boost_round=rounds, verbose_eval=False)

    history = train[["date","store","item","sales"]].copy()
    history["date"] = pd.to_datetime(history.date)
    out = te.copy()
    out["sales_prediction"] = np.nan
    for day in sorted(out.date.dropna().unique()):
        mask = out.date.eq(day)
        rows = out.loc[mask].copy()
        lookup = history.set_index(["date","store","item"]).sales
        for lag in [7,14,28,365]:
            dates = rows.date - pd.Timedelta(days=lag)
            keys = pd.MultiIndex.from_arrays([dates,rows.store,rows.item],
                                             names=["date","store","item"])
            rows[f"lag_{lag}"] = lookup.reindex(keys).to_numpy()
        dday = xgb.DMatrix(rows[FEATURES].fillna(0), feature_names=FEATURES)
        pred = np.clip(np.expm1(final_model.predict(dday)), 0, None)
        out.loc[mask, "sales_prediction"] = pred
        added = rows[["date","store","item"]].copy()
        added["sales"] = pred
        history = pd.concat([history,added], ignore_index=True)
    return out, model_score, baseline_score, final_model, len(full)

with st.sidebar:
    st.header("Forecast settings")
    train_file = st.file_uploader("Upload train.csv", type="csv")
    test_file = st.file_uploader("Upload test.csv", type="csv")
    rounds = st.slider("Boosting rounds", 100, 700, 300, 50)
    eta = st.select_slider("Learning rate", options=[0.02,0.03,0.05,0.08,0.1], value=0.05)
    depth = st.slider("Tree depth", 3, 10, 6)
    run = st.button("🚀 Train & forecast", type="primary", use_container_width=True)

st.markdown("""
**Training CSV** requires `date`, `store`, `item`, `sales`.  
**Test CSV** requires `date`, `store`, `item`.  
Upload the two CSV files from the sidebar, then run the forecast.
""")

if run:
    if train_file is None or test_file is None:
        st.error("Please upload both train.csv and test.csv.")
    else:
        try:
            train = pd.read_csv(train_file)
            test = pd.read_csv(test_file)
            required_train = {"date","store","item","sales"}
            required_test = {"date","store","item"}
            if not required_train.issubset(train.columns):
                raise ValueError(f"Training file needs columns: {sorted(required_train)}")
            if not required_test.issubset(test.columns):
                raise ValueError(f"Test file needs columns: {sorted(required_test)}")
            for df in [train,test]:
                df["date"] = pd.to_datetime(df.date, errors="coerce")
                df["store"] = pd.to_numeric(df.store, errors="coerce")
                df["item"] = pd.to_numeric(df.item, errors="coerce")
            train["sales"] = pd.to_numeric(train.sales, errors="coerce")
            train = train.dropna(subset=["date","store","item","sales"])
            test = test.dropna(subset=["date","store","item"])
            with st.spinner("Training model and generating recursive forecasts..."):
                result, score, baseline, model, n_model = forecast_sales(train,test,rounds,eta,depth)
            st.session_state["result"] = result
            st.session_state["metrics"] = (len(train),len(test),train.store.nunique(),
                                           train.item.nunique(),score,baseline,n_model)
            st.session_state["model"] = model
            st.success("Forecast generated.")
        except Exception as e:
            st.error(f"Forecast failed: {e}")

if "result" in st.session_state:
    result = st.session_state["result"]
    ntrain, ntest, nstores, nitems, score, baseline, n_model = st.session_state["metrics"]
    st.subheader("Forecast overview")
    cols = st.columns(4)
    cols[0].metric("Training rows", f"{ntrain:,}")
    cols[1].metric("Stores", f"{nstores:,}")
    cols[2].metric("Items", f"{nitems:,}")
    cols[3].metric("Forecast rows", f"{ntest:,}")
    cols = st.columns(3)
    cols[0].metric("Validation sMAPE", f"{score:.2f}%")
    cols[1].metric("Mean baseline sMAPE", f"{baseline:.2f}%")
    cols[2].metric("Difference vs baseline", f"{baseline-score:+.2f} pp")
    st.caption(f"Final model trained on {n_model:,} rows. Validation is an initial diagnostic, not a perfect multi-step backtest.")
    tab1, tab2, tab3 = st.tabs(["Charts","Predictions","Feature importance"])
    with tab1:
        daily = result.groupby("date", as_index=False).sales_prediction.sum()
        st.plotly_chart(px.line(daily, x="date", y="sales_prediction", title="Predicted total sales by day"), use_container_width=True)
        stores = result.groupby("store", as_index=False).sales_prediction.sum()
        st.plotly_chart(px.bar(stores, x="store", y="sales_prediction", title="Forecast demand by store"), use_container_width=True)
        items = result.groupby("item", as_index=False).sales_prediction.sum()
        st.plotly_chart(px.bar(items, x="item", y="sales_prediction", title="Forecast demand by item"), use_container_width=True)
    with tab2:
        stores = sorted(result.store.unique().tolist())
        items = sorted(result.item.unique().tolist())
        c1,c2 = st.columns(2)
        selected_stores = c1.multiselect("Stores", stores, default=stores[:min(5,len(stores))])
        selected_items = c2.multiselect("Items", items, default=items[:min(10,len(items))])
        filtered = result[result.store.isin(selected_stores) & result.item.isin(selected_items)]
        st.dataframe(filtered[["date","store","item","sales_prediction"]], use_container_width=True)
        st.download_button("⬇️ Download predictions CSV",
            result[["date","store","item","sales_prediction"]].to_csv(index=False).encode(),
            "store_demand_forecast.csv","text/csv",type="primary")
    with tab3:
        imp = st.session_state["model"].get_score(importance_type="gain")
        imp_df = pd.DataFrame([{"feature":k,"gain":v} for k,v in imp.items()]).sort_values("gain")
        if len(imp_df):
            st.plotly_chart(px.bar(imp_df,x="gain",y="feature",orientation="h",
                                   title="XGBoost feature importance (gain)"),use_container_width=True)
        else:
            st.info("No feature importance values available.")
    st.info("Business use: combine forecasts with current inventory, supplier lead times, and promotions before making replenishment decisions.")
