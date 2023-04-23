# -*- coding: utf-8 -*-
"""
Author:
    Xuxin Zhang,xuxinz@qq.com
Reference: Chae D K , Kang J S , Kim S W , et al.
CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks[C]// the 27th ACM International Conference. ACM, 2018.

"""

from collections import defaultdict
import torch
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def loadUseInfo(trainFile, splitMark):
    print(trainFile)
    UseInfo = pd.DataFrame(
        columns=["userId", "useAge", "useGender", "merchantId", "category"])
    index = 0
    for line in open(trainFile):
        userId, useAge, useGender, merchantId, category = line.strip().split(splitMark)
        UseInfo.loc['%d' % index] = [
            userId, useAge, useGender, merchantId, category]
        index = index + 1
    UseInfo.to_csv("userInfo.csv", index=False)
    print("Demographic information about the users loading done: %d users" % (index))
    return UseInfo


def loadItemInfo(trainFile, splitMark):
    #    trainFile = "data/ml-100k/u.item"
    #    splitMark =  "|"
    print(trainFile)
    ItemInfo = pd.DataFrame(columns=["movie_id", "movie_title", "release_date", "video_release_date",
                                     "IMDb_URL", "unknowngenres", "Action", "Adventure",
                                     "Animation", "Childrens", "Comedy", "Crime",
                                     "Documentary", "Drama", "Fantasy", "Film_Noir",
                                     "Horror", "Musical", "Mystery", "Romance",
                                     "Sci_Fi", "Thriller", "War", "Western"])
    index = 0
    for line in open(trainFile, encoding='ISO-8859-1'):
        ItemInfo.loc['%d' % index] = line.strip().split(splitMark)
        index = index + 1
    ItemInfo.to_csv("data/ml-100k/itemInfo.csv")
    print("Information about the items loading done: %d users" % (index))
    return ItemInfo


def loadTrainingData(trainFile, splitMark):
    trainSet = defaultdict(list)
    max_u_id = -1
    max_i_id = -1
    for line in open(trainFile):
        userId, itemId, rating, _, _ = line.strip().split(splitMark)
        userId = int(userId)
        itemId = int(itemId)
        trainSet[userId].append(itemId)
        max_u_id = max(userId, max_u_id)
        max_i_id = max(itemId, max_i_id)
    userCount = max_u_id + 1
    itemCount = max_i_id + 1
    print("Training data loading done")
    return trainSet, userCount, itemCount


def loadTestData(testFile, splitMark):
    testSet = defaultdict(list)
    max_u_id = -1
    max_i_id = -1
    for line in open(testFile):
        userId, itemId, rating, _, _ = line.strip().split(splitMark)
        userId = int(userId)
        itemId = int(itemId)
        testSet[userId].append(itemId)
        max_u_id = max(userId, max_u_id)
        max_i_id = max(itemId, max_i_id)
    userCount = max_u_id + 1
    itemCount = max_i_id + 1
    print("Test data loading done")
    return testSet, userCount, itemCount


def to_Vectors(trainSet, userCount, itemCount, userList_test, mode):

    testMaskDict = defaultdict(lambda: [0] * itemCount)
    batchCount = userCount  # 改动  直接写成userCount
    if mode == "itemBased":  # 改动  itemCount userCount互换   batchCount是物品数
        userCount = itemCount
        itemCount = batchCount
        batchCount = userCount
    trainDict = defaultdict(lambda: [0] * itemCount)
    for userId, i_list in trainSet.items():
        for itemId in i_list:
            testMaskDict[userId][itemId] = -99999
            if mode == "userBased":
                trainDict[userId][itemId] = 1.0
            else:
                trainDict[itemId][userId] = 1.0

    trainVector = []
    for batchId in range(batchCount):
        trainVector.append(trainDict[batchId])

    testMaskVector = []
    for userId in userList_test:
        testMaskVector.append(testMaskDict[userId])
    print("Converting to vectors done....")
    return (torch.Tensor(trainVector)), torch.Tensor(testMaskVector), batchCount


def pre_process_raw_data(input_file='raw.txt', output_file='welldone'):
    dir_name = f'./welldones'
    os.makedirs(dir_name, exist_ok=True)
    output_file = f'{dir_name}/{output_file}'

    # open the input and output files
    ss = []
    with open(input_file, 'r') as infile:
        # iterate over each line in the input file
        for line in infile:
            # split the line into fields
            fields = line.strip().split()[:-1]
            if '9999' not in fields:
                # write the fields to the CSV file
                ss.append(fields)
            else:
                print(f"drop: {fields}")
    df = pd.DataFrame(
        ss, columns='UserId, gender, age, merchantId, category'.split(', '))
    new_order = ["UserId", "merchantId", "gender", "age", "category"]
    df = df[new_order]

    # create a mapping of UserId to sequential integer IDs
    user_mapping = {user_id: i+1 for i,
                    user_id in enumerate(df['UserId'].unique())}

    # create a mapping of MerchantId to sequential integer IDs
    merchant_mapping = {merchant_id: i+1 for i,
                        merchant_id in enumerate(df.iloc[:, 1].unique())}

    num_item = len(df.iloc[:, 1].unique())

    # replace UserId and merchantId with their corresponding integer IDs
    df['UserId'] = df['UserId'].map(user_mapping)
    df["merchantId"] = df["merchantId"].map(merchant_mapping)

    # write the transformed data to a new CSV file

    df.to_csv(f"./{output_file}.csv", index=False)

    saved_user = set()
    with open(f'./{output_file}.user', 'w') as f:
        for s in df.values:
            if s[0] not in saved_user:
                n = [s[0], int(s[3]), s[2], int(0), int(0)]
                ss = ''.join(str(i)+'|' for i in n)
                f.write(ss[:-1] + '\n')
                saved_user.add(s[0])

    # slice to u*.base, u*.test
    # mkdir
    u_dir_name = f"{dir_name}/u"
    os.makedirs(u_dir_name, exist_ok=True)

    for i in range(1, 6):
        train_data, test_data = train_test_split(
            df, test_size=0.2, random_state=i)
        train_data.to_csv(f"{u_dir_name}/u{i}.base",
                          index=False, header=False, sep=',')
        test_data.to_csv(f"{u_dir_name}/u{i}.test",
                         index=False, header=False, sep=',')

    return dir_name, num_item+1
