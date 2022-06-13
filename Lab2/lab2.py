{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "792ccfa2-df88-4e99-8389-0db1042db546",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.datasets import fetch_openml \n",
    "mnist = fetch_openml('mnist_784', version=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "70291fa4-f54d-47bf-bac3-a2e194f5dd49",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "print((np.array(mnist.data.loc[42]).reshape(28, 28) > 0).astype(int))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "46e1ef56-a22e-4f54-9eb1-607d99da28ff",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = mnist.data\n",
    "y = mnist.target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ebc2bf8a-b7c2-41c2-99a9-fe17f6ee441b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "y = y.sort_values(ascending=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8d4dbbb4-3bb8-4c62-a7cc-25e0eaac119b",
   "metadata": {},
   "outputs": [],
   "source": [
    "y.index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c3b8322c-071d-425f-9e24-af2363e2cc21",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = X.reindex(y.index)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "24fbba45-68a9-4d80-9c07-52ac487c3509",
   "metadata": {},
   "outputs": [],
   "source": [
    "X.index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f420f3fc-db28-4a01-9a67-1d795bcca6d3",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test = X[:56000], X[56000:]\n",
    "y_train, y_test = y[:56000], y[56000:]\n",
    "print(X_train.shape, y_train.shape)\n",
    "print(X_test.shape, y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "73098bd8-3d7f-44ca-a89f-827b753019e9",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train.unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0ea4f4fc-be88-4a45-a82b-a9056b3d4de2",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_test.unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5ea1deb1-34c2-4160-bc4f-97e4673f64b3",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "535172b8-6f46-4813-8f97-93ec009c298c",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train.unique().sort_values()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "217c0fec-084a-4024-817a-38246854c671",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_test.unique().sort_values()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2b00c7dd-8b02-4f91-85e5-4b9428659382",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e20fa569-18df-4d58-b1d3-9f37f3d7d2e3",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train_0 = (y_train == '0')\n",
    "y_test_0 = (y_test == '0')\n",
    "print(y_train_0)\n",
    "print(np.unique(y_train_0))\n",
    "print(len(y_train_0))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5f771049-c784-4a80-8471-29968e1ea28e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d8322e64-065f-4b82-b1a8-ffdd20986996",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5ed4d00e-8ecf-4136-ae33-dbca1b4753fc",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import SGDClassifier\n",
    "import time\n",
    "start = time.time()\n",
    "sgd_clf = SGDClassifier(random_state=42)\n",
    "sgd_clf.fit(X_train, y_train_0)\n",
    "print(time.time() - start)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c5d23bb9-fdc5-4511-bb44-9914eda62f67",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import cross_val_score\n",
    "import pickle\n",
    "start = time.time()\n",
    "score = cross_val_score(sgd_clf, X_train, y_train_0,\n",
    "                        cv=3, scoring=\"accuracy\",\n",
    "                        n_jobs=-1)\n",
    "print(time.time() - start)\n",
    "print(score)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "931927b7-7f3e-45e9-99a9-4704b39f3aba",
   "metadata": {},
   "outputs": [],
   "source": [
    "pickle.dump(score,open('sgd_cva.pkl', 'wb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "539a9625-0df3-487c-9da8-28ac0c0444ad",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import cross_val_predict\n",
    "from sklearn.metrics import confusion_matrix\n",
    "y_train_pred = cross_val_predict(sgd_clf, X_train,y_train_0, cv=3,n_jobs=-1)\n",
    "y_test_pred = cross_val_predict(sgd_clf, X_test,y_test_0, cv=3,n_jobs=-1)\n",
    "print(y_train_pred)\n",
    "print(len(y_train_pred))\n",
    "print(confusion_matrix(y_train_0, y_train_pred))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "afddc3a4-6228-4e17-97e7-a78cf7686403",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(confusion_matrix(y_train_0, y_train_0))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "09705afd-cae7-4a2a-9fdd-21e11bc996f2",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(confusion_matrix(y_test_0, y_test_0))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e70d1c85-ee59-4739-a64c-aaac89a5ccae",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import precision_score, recall_score\n",
    "print(precision_score(y_train_0, y_train_pred))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6ea81623-7de8-4015-8d78-63e8a5a0d535",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(precision_score(y_test_0, y_test_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "da5f337e-6397-4aeb-8d16-8f60e4d5bf3e",
   "metadata": {},
   "outputs": [],
   "source": [
    "list1 = [precision_score(y_test_0, y_test_pred),precision_score(y_train_0, y_train_pred)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5693b99e-4b14-47b6-98db-8831c140b80f",
   "metadata": {},
   "outputs": [],
   "source": [
    "pickle.dump(list1,open('sgd_acc.pkl', 'wb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e0ab91ab-bec4-4c89-b043-4395b2e36802",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c7129b3b-c9f4-465d-ab71-8769e0955ec5",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "sgd_m_clf = SGDClassifier(random_state=42,n_jobs=-1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "913a33d4-e0e8-4eff-a85c-13b5b47722fe",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(cross_val_score(sgd_m_clf, X_train, y_train, cv=3, scoring=\"accuracy\", n_jobs=-1))\n",
    "y_train_pred = cross_val_predict(sgd_m_clf, X_train, y_train, cv=3, n_jobs=-1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "552cd123-887d-4668-9931-88a0e5f66268",
   "metadata": {},
   "outputs": [],
   "source": [
    "conf_mx = confusion_matrix(y_train, y_train_pred)\n",
    "print(conf_mx)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b7307966-a4da-435b-accc-0c56c3803656",
   "metadata": {},
   "outputs": [],
   "source": [
    "pickle.dump(conf_mx,open('sgd_cmx.pkl','wb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "26bf9a93-2c0f-4346-8305-871e748d78fb",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "row_sums = conf_mx.sum(axis=1, keepdims=True)\n",
    "norm_conf_mx = conf_mx / row_sums\n",
    "np.fill_diagonal(norm_conf_mx, 0) # zeby nie przeszkadzalo\n",
    "plt.matshow(norm_conf_mx, cmap=plt.cm.gray)\n",
    "f = \"norm_conf_mx.png\"\n",
    "plt.savefig(f)\n",
    "print(f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cd7ca8fd-1384-4c86-af58-399208ff809f",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
