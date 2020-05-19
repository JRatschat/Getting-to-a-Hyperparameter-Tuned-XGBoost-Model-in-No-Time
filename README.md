## About the repository
This blog post was written in summer 2020.

## Short primer on the blog post
Let's face it. You're an aspiring Data Scientist and you need to desperately solve a classification task either to impress your professor or to finally become one of these mysterious Kaggle winners. The problem is that you've got no time. The presentation or the Kaggle competition is due in 24 hours. Of course, you could stick to a simple logit regression and call it a day. "No, not this time!" you think and you're browsing the web to find THE SOLUTION. But it better be fast.

Okay, this an highly improbable event. However, I wanted to share an easy XGBoost implementation that proved to be lightning fast (compared to other solutions) that can definitely help you to achieve a more stable and accurate model. 
Looking back, hyperparameter-tuning was the step in which I lost most of my "valuable" time; therefore, the focus will be on this step of building an XGBoost model.

Why are we using XGBoost again? Because it rocks. It has been used with great success on many machine learning challenges. Among the advantages of XGBoost are that its models are easily scalable, tend to avoid overfitting, and can be used for a wide range of problems.

Of course, only preparing the data and executing XGBoost would somehow work. But clearly, there is no free lunch in data science. This procedure would leave you with a model that is most likely overfitted and therefore performs bad on an out-of-sample data set. Meaning: you have created an useless model that you should not use with other data.

Luckily, XGBoost offers several ways to avoid overfitting and to make sure that the performance of the model is optimized. Hyperparameter-tuning is the last part of the model building and can definitely have an impact on your model's performance. Tuning is a systematic and automated process of varying parameters to find the "best" model.

Read the rest of this blog post here: [Getting to an Hyperparameter-Tuned XGBoost Model in No Time](https://medium.com/@j.ratschat/getting-to-an-hyperparameter-tuned-xgboost-model-in-no-time-a9560f8eb54b?source=friends_link&sk=20cf093923ba6f55945e432c2a5c8e6a)
