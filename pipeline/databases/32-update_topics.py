#!/usr/bin/env python3
"""Module that updates topics of a school document"""


def update_topics(mongo_collection, name, topics):
    """Updates the topics of a school based on the name"""
    mongo_collection.update_many(
        {"name": name},
        {"$set": {"topics": topics}}
    )
