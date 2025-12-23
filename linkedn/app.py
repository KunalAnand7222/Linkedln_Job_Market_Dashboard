# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go

# st.set_page_config(page_title="LinkedIn Job Market Dashboard",layout="wide",initial_sidebar_state="expanded")

# ku=pd.read_csv("postings.csv")

# st.markdown("""
# <style>
# body{background-color:#0e1117}
# .block-container{padding-top:1.5rem}
# h1,h2,h3,h4{color:white}
# [data-testid=stSidebar]{background-color:#111827}
# </style>
# """,unsafe_allow_html=True)

# st.sidebar.markdown("## 🔍 Filters")

# loc=st.sidebar.selectbox("Location",["All"]+sorted(ku['location'].dropna().unique().tolist()))
# wrk=st.sidebar.selectbox("Work Type",["All"]+sorted(ku['formatted_work_type'].dropna().unique().tolist()))
# exp=st.sidebar.selectbox("Experience Level",["All"]+sorted(ku['formatted_experience_level'].dropna().unique().tolist()))

# df=ku.copy()
# if loc!="All":
#     df=df[df['location']==loc]
# if wrk!="All":
#     df=df[df['formatted_work_type']==wrk]
# if exp!="All":
#     df=df[df['formatted_experience_level']==exp]

# st.markdown("# 💼 LinkedIn Job Market Analysis")
# st.markdown("### Hiring trends, salaries & skill demand")

# c1,c2,c3,c4=st.columns(4)
# with c1:
#     st.metric("Total Jobs",len(df))
# with c2:
#     st.metric("Companies",df['company_name'].nunique())
# with c3:
#     st.metric("Locations",df['location'].nunique())
# with c4:
#     st.metric("Avg Salary",int(df['normalized_salary'].dropna().mean()) if df['normalized_salary'].notna().any() else 0)

# st.markdown("---")

# st.markdown("## 📊 Hiring Demand Overview")

# l1,l2=st.columns(2)

# with l1:
#     xx=df['title'].value_counts().head(12)
#     fig1=px.bar(
#         x=xx.values,
#         y=xx.index,
#         orientation='h',
#         color=xx.values,
#         color_continuous_scale='blues',
#         title="Top Job Roles"
#     )
#     fig1.update_layout(height=450,xaxis_title="Openings",yaxis_title="")
#     st.plotly_chart(fig1,use_container_width=True)

# with l2:
#     xy=df['location'].value_counts().head(12)
#     fig2=px.bar(
#         x=xy.values,
#         y=xy.index,
#         orientation='h',
#         color=xy.values,
#         color_continuous_scale='teal',
#         title="Top Hiring Locations"
#     )
#     fig2.update_layout(height=450,xaxis_title="Jobs",yaxis_title="")
#     st.plotly_chart(fig2,use_container_width=True)

# st.markdown("---")

# st.markdown("## 🧭 Work & Experience Trends")

# m1,m2=st.columns(2)

# with m1:
#     wt=df['formatted_work_type'].value_counts()
#     fig3=px.pie(
#         names=wt.index,
#         values=wt.values,
#         hole=0.45,
#         title="Work Type Distribution",
#         color_discrete_sequence=px.colors.sequential.Plasma
#     )
#     st.plotly_chart(fig3,use_container_width=True)

# with m2:
#     ex=df['formatted_experience_level'].value_counts()
#     fig4=px.bar(
#         x=ex.index,
#         y=ex.values,
#         color=ex.values,
#         color_continuous_scale='sunset',
#         title="Experience Level Demand"
#     )
#     fig4.update_layout(xaxis_title="",yaxis_title="Jobs")
#     st.plotly_chart(fig4,use_container_width=True)

# st.markdown("---")

# st.markdown("## 🌍 Remote Work Insights")

# r1,r2=st.columns(2)

# with r1:
#     ra=df['remote_allowed'].value_counts().reset_index()
#     ra.columns=['remote','count']
#     ra['remote']=ra['remote'].map({0:'Not Remote',1:'Remote'})
#     fig5=px.pie(
#         ra,
#         names='remote',
#         values='count',
#         hole=0.5,
#         title="Remote Allowed Jobs",
#         color_discrete_sequence=["#636EFA","#00CC96"]
#     )
#     st.plotly_chart(fig5,use_container_width=True)

# with r2:
#     df['listed_time']=pd.to_datetime(df['listed_time'],errors='coerce')
#     ts=df.groupby(df['listed_time'].dt.to_period('M')).size()
#     fig6=go.Figure()
#     fig6.add_trace(go.Scatter(x=ts.index.astype(str),y=ts.values,mode='lines+markers'))
#     fig6.update_layout(title="Job Posting Trend Over Time",xaxis_title="Month",yaxis_title="Jobs")
#     st.plotly_chart(fig6,use_container_width=True)
# st.markdown("---")
# st.markdown("## 💰 Salary Insights")

# s1,s2=st.columns(2)

# with s1:
#     df_sal=df.dropna(subset=['normalized_salary','formatted_experience_level','formatted_work_type'])
#     fig7=px.box(
#         df_sal,
#         x='formatted_experience_level',
#         y='normalized_salary',
#         color='formatted_experience_level',
#         title="Salary vs Experience Level",
#         color_discrete_sequence=px.colors.qualitative.Bold
#     )
#     fig7.update_layout(xaxis_title="",yaxis_title="Salary")
#     st.plotly_chart(fig7,use_container_width=True)

# with s2:
#     fig8=px.violin(
#         df_sal,
#         x='formatted_work_type',
#         y='normalized_salary',
#         color='formatted_work_type',
#         box=True,
#         points='outliers',
#         title="Salary Distribution by Work Type",
#         color_discrete_sequence=px.colors.sequential.Agsunset
#     )
#     fig8.update_layout(xaxis_title="",yaxis_title="Salary")
#     st.plotly_chart(fig8,use_container_width=True)

# st.markdown("---")
# st.markdown("## 🧠 Skill Demand Analysis")

# df['sklen']=df['skills_desc'].fillna("").apply(lambda x:len(x.split(",")))

# k1,k2=st.columns(2)

# with k1:
#     from sklearn.feature_extraction.text import CountVectorizer
#     cv=CountVectorizer(stop_words='english',max_features=20)
#     mat=cv.fit_transform(df['skills_desc'].dropna().astype(str))
#     ab=pd.DataFrame(mat.toarray(),columns=cv.get_feature_names_out())
#     sk=ab.sum().reset_index()
#     sk.columns=['skill','count']
#     fig9=px.treemap(
#         sk,
#         path=['skill'],
#         values='count',
#         title="Top In-Demand Skills",
#         color='count',
#         color_continuous_scale='viridis'
#     )
#     st.plotly_chart(fig9,use_container_width=True)

# with k2:
#     df_b=df.dropna(subset=['normalized_salary','sklen'])
#     fig10=px.scatter(
#         df_b,
#         x='sklen',
#         y='normalized_salary',
#         size='sklen',
#         color='formatted_experience_level',
#         title="Skills vs Salary Bubble Chart",
#         labels={'sklen':'Number of Skills','normalized_salary':'Salary'},
#         color_discrete_sequence=px.colors.qualitative.Set2
#     )
#     st.plotly_chart(fig10,use_container_width=True)

# st.markdown("---")
# st.markdown("## 🤖 Machine Learning: Salary Prediction")

# ml=df[['normalized_salary','formatted_experience_level','formatted_work_type','remote_allowed']].dropna()
# ml['remote_allowed']=ml['remote_allowed'].astype(int)
# X=pd.get_dummies(ml.drop('normalized_salary',axis=1))
# y=ml['normalized_salary']

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score

# Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
# rf=RandomForestRegressor(n_estimators=120,max_depth=10,random_state=42)
# rf.fit(Xtr,ytr)
# yp=rf.predict(Xte)

# c1,c2=st.columns(2)
# with c1:
#     st.metric("Model R² Score",round(r2_score(yte,yp),3))
# with c2:
#     st.metric("Training Samples",len(Xtr))

# st.markdown("### 🎯 Predict Salary")

# e1=st.selectbox("Experience Level",X.columns[X.columns.str.contains('formatted_experience_level')])
# w1=st.selectbox("Work Type",X.columns[X.columns.str.contains('formatted_work_type')])
# r1=st.selectbox("Remote Allowed",[0,1])

# ip=pd.DataFrame(0,index=[0],columns=X.columns)
# ip[e1]=1
# ip[w1]=1
# ip['remote_allowed']=r1

# st.success(f"Predicted Salary: ${int(rf.predict(ip)[0]):,}")
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="LinkedIn Job Market Dashboard",layout="wide",initial_sidebar_state="expanded")

post=pd.read_csv("postings.csv")
jobsal=pd.read_csv("salaries.csv")
comp=pd.read_csv("companies.csv")
emp=pd.read_csv("employee_counts.csv")

st.markdown("""
<style>
body{background-color:#0e1117}
.block-container{padding-top:1.5rem}
h1,h2,h3,h4{color:white}
[data-testid=stSidebar]{background-color:#111827}
</style>
""",unsafe_allow_html=True)

post=post.merge(jobsal,on="job_id",how="left",suffixes=("","_s"))
post=post.merge(comp,on="company_id",how="left")
post=post.merge(
    emp.groupby("company_id")[["employee_count","follower_count"]]
    .max()
    .reset_index(),
    on="company_id",
    how="left"
)

st.sidebar.markdown("## 🔍 Filters")

loc=st.sidebar.selectbox("Location",["All"]+sorted(post["location"].dropna().unique()))
wrk=st.sidebar.selectbox("Work Type",["All"]+sorted(post["formatted_work_type"].dropna().unique()))
exp=st.sidebar.selectbox("Experience Level",["All"]+sorted(post["formatted_experience_level"].dropna().unique()))

df=post.copy()
if loc!="All":
    df=df[df["location"]==loc]
if wrk!="All":
    df=df[df["formatted_work_type"]==wrk]
if exp!="All":
    df=df[df["formatted_experience_level"]==exp]

st.markdown("# 💼 LinkedIn Job Market Dashboard")
st.markdown("### Hiring trends, salaries & workforce insights")

c1,c2,c3,c4=st.columns(4)
c1.metric("Total Jobs",len(df))
c2.metric("Companies",df["company_id"].nunique())
c3.metric("Locations",df["location"].nunique())
c4.metric("Avg Salary",int(df["normalized_salary"].dropna().mean()) if df["normalized_salary"].notna().any() else 0)

st.markdown("---")
st.markdown("## 📊 Hiring Demand Overview")

l1,l2=st.columns(2)

jr=df["title"].value_counts().head(12)
fig1=px.bar(
    x=jr.values,
    y=jr.index,
    orientation="h",
    color=jr.values,
    color_continuous_scale="blues",
    title="Top Job Roles"
)
l1.plotly_chart(fig1,use_container_width=True)

lc=df["location"].value_counts().head(12)
fig2=px.bar(
    x=lc.values,
    y=lc.index,
    orientation="h",
    color=lc.values,
    color_continuous_scale="teal",
    title="Top Hiring Locations"
)
l2.plotly_chart(fig2,use_container_width=True)

st.markdown("---")
st.markdown("## 🧭 Work & Experience Trends")

w1,w2=st.columns(2)

with w1:
    wt=df["formatted_work_type"].value_counts().reset_index()
    wt.columns=["type","count"]
    fig3=px.pie(
        wt,
        names="type",
        values="count",
        hole=0.6,
        title="Work Type Distribution",
        color_discrete_sequence=[
            "#1f1fd1",
            "#6a0dad",
            "#9b4dcc",
            "#c77dff",
            "#ff6fb1",
            "#ff9f80",
            "#ffa07a"
        ]
    )
    fig3.update_traces(textposition="outside",textinfo="percent")
    st.plotly_chart(fig3,use_container_width=True)

with w2:
    ex=df["formatted_experience_level"].value_counts().reset_index()
    ex.columns=["level","count"]
    fig4=px.bar(
        ex,
        x="level",
        y="count",
        color="count",
        title="Experience Level Demand",
        color_continuous_scale=[
            "#fef3c7",
            "#fde68a",
            "#fcd34d",
            "#fbbf24",
            "#f59e0b",
            "#d97706"
        ]
    )
    fig4.update_layout(xaxis_title="",yaxis_title="Jobs",xaxis_tickangle=-30)
    st.plotly_chart(fig4,use_container_width=True)

st.markdown("---")
st.markdown("## 🌍 Job Activity")

a1,a2=st.columns(2)

ra=df["remote_allowed"].value_counts().reset_index()
ra.columns=["remote","count"]
ra["remote"]=ra["remote"].map({0:"Not Remote",1:"Remote"})
fig5=px.pie(
    ra,
    names="remote",
    values="count",
    hole=0.5,
    title="Remote vs Onsite Jobs",
    color_discrete_sequence=["#636EFA","#00CC96"]
)
a1.plotly_chart(fig5,use_container_width=True)

df["listed_time"]=pd.to_datetime(df["listed_time"],errors="coerce")
ts=df.groupby(df["listed_time"].dt.to_period("M")).size()
fig6=go.Figure(go.Scatter(x=ts.index.astype(str),y=ts.values,mode="lines+markers"))
fig6.update_layout(title="Job Posting Trend Over Time",xaxis_title="Month",yaxis_title="Jobs")
a2.plotly_chart(fig6,use_container_width=True)


st.markdown("---")
st.markdown("## 🧠 Skills & Industry Insights")

jobskill=pd.read_csv("job_skills.csv")
skills=pd.read_csv("skills.csv")
jobind=pd.read_csv("job_industries.csv")
inds=pd.read_csv("industries.csv")

sk=jobskill.merge(skills,on="skill_abr",how="left")
sk=sk[sk["job_id"].isin(df["job_id"])]

top_sk=sk["skill_name"].value_counts().head(20).reset_index()
top_sk.columns=["skill","count"]

fig7=px.treemap(
    top_sk,
    path=["skill"],
    values="count",
    color="count",
    color_continuous_scale="viridis",
    title="Top In-Demand Skills"
)
st.plotly_chart(fig7,use_container_width=True)

ji=jobind.merge(inds,on="industry_id",how="left")
ji=ji[ji["job_id"].isin(df["job_id"])]

top_ind=ji["industry_name"].value_counts().head(15).reset_index()
top_ind.columns=["industry","count"]

fig8=px.bar(
    top_ind,
    x="count",
    y="industry",
    orientation="h",
    color="count",
    color_continuous_scale="plasma",
    title="Top Hiring Industries"
)
st.plotly_chart(fig8,use_container_width=True)
st.markdown("---")
st.markdown("## 💰 Salary Analysis")

sal=df.dropna(subset=["normalized_salary"])
sal=sal[sal["normalized_salary"]<300000]

s1,s2=st.columns(2)

fig9=px.box(
    sal,
    x="formatted_experience_level",
    y="normalized_salary",
    color="formatted_experience_level",
    title="Salary vs Experience Level"
)
s1.plotly_chart(fig9,use_container_width=True)

fig10=px.violin(
    sal,
    x="formatted_work_type",
    y="normalized_salary",
    color="formatted_work_type",
    box=True,
    points="outliers",
    title="Salary Distribution by Work Type"
)
s2.plotly_chart(fig10,use_container_width=True)

st.markdown("---")
st.markdown("## 📈 Skills vs Salary Relationship")

skill_cnt=sk.groupby("job_id").size().reset_index(name="skill_count")
df2=df.merge(skill_cnt,on="job_id",how="left")

fig11=px.scatter(
    df2.dropna(subset=["normalized_salary","skill_count"]),
    x="skill_count",
    y="normalized_salary",
    size="skill_count",
    color="formatted_experience_level",
    title="Skill Count vs Salary",
    labels={
        "skill_count":"Number of Skills",
        "normalized_salary":"Salary"
    }
)
st.plotly_chart(fig11,use_container_width=True)

st.markdown("---")
st.markdown("## 🤖 Machine Learning: Salary Prediction")

ml=df2[
    [
        "normalized_salary",
        "skill_count",
        "remote_allowed",
        "formatted_experience_level",
        "formatted_work_type"
    ]
].dropna()

ml=ml[ml["normalized_salary"]<300000]

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

y=np.log1p(ml["normalized_salary"])
X=pd.get_dummies(ml.drop("normalized_salary",axis=1))

Xtr,Xte,ytr,yte=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

rf=RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42
)
rf.fit(Xtr,ytr)
yp=rf.predict(Xte)

m1,m2=st.columns(2)
m1.metric("Model R² Score",round(r2_score(yte,yp),3))
m2.metric("Training Samples",len(Xtr))

st.markdown("### 🎯 Predict Salary")

e=st.selectbox(
    "Experience Level",
    [c for c in X.columns if "formatted_experience_level" in c]
)
w=st.selectbox(
    "Work Type",
    [c for c in X.columns if "formatted_work_type" in c]
)
s=st.slider("Skill Count",1,50,10)
r=st.selectbox("Remote Allowed",[0,1])

ip=pd.DataFrame(0,index=[0],columns=X.columns)
ip[e]=1
ip[w]=1
ip["skill_count"]=s
ip["remote_allowed"]=r

pred=int(np.expm1(rf.predict(ip))[0])
st.success(f"Predicted Salary: ${pred:,}")
