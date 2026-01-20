import json
import os
from typing import Dict, List, Any


class BenchmarkDataLoader:
    """
    数据加载器类，用于加载 benchmark JSON 数据
    """

    def __init__(self, data_path: str = None):
        """
        初始化数据加载器

        Args:
            data_path: 数据文件路径，如果为 None 则使用默认路径
        """
        if data_path is None:
            # 使用基于当前文件位置的相对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_path = os.path.join(current_dir, "benchmark_output", "benchmark_3.0.json")
        else:
            self.data_path = data_path

        self.data = None

    def run(self) -> Dict[str, Any]:
        """
        读取并加载 benchmark JSON 文件

        Returns:
            包含 benchmark 数据的字典
        """
        # 读取 JSON 文件
        print(f"正在加载数据文件: {self.data_path}")
        with open(self.data_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)

        print(f"数据加载成功! 包含 {len(self.data.get('review_groups', []))} 个 review groups")
        print("加载reviewGroups数据...")
        self.reviewAndQuery = self.get_all_reviews_and_queries()
        total_reviews = sum(len(group['reviews']) for group in self.reviewAndQuery)
        total_hypotheses = sum(len(group['queries']) for group in self.reviewAndQuery)
        print(f"reviewGroups数据加载成功! 包含 {total_reviews} 个 review")
        print(f"reviewGroups数据加载成功! 包含 {total_hypotheses} 个 hypotheses")
        return self.reviewAndQuery

    def get_review_groups(self) -> List[Dict[str, Any]]:
        """
        获取所有 review groups

        Returns:
            review groups 列表
        """
        return self.data.get('review_groups', [])

    def get_reviews_count(self) -> int:
        """
        获取总的 reviews 数量

        Returns:
            reviews 总数
        """
        count = 0
        for group in self.data.get('review_groups', []):
            count += len(group.get('reviews', []))
        return count

    def get_all_reviews_and_queries(self) -> List[Dict[str, Any]]:
        """
        遍历所有 review_groups，将每个 group 的 reviews 和 query 分别存起来

        Returns:
            包含所有 reviews 和对应 query 的数据列表
            每个元素包含: group_id, query (topic), reviews, hypotheses
        """
        all_data = []
        review_groups = self.data.get('review_groups', [])

        for group in review_groups:
            group_data = {
                'group_id': group.get('group_id'),
                'topic': group.get('topic'),  # 使用 topic 作为 query
                'reviews': group.get('reviews', []),
                'hypotheses': group.get('query', {}).get('hypotheses', []),  # 获取 query 下的 hypotheses
                'queries': group.get('query', {}).get('queries',[])
            }
            all_data.append(group_data)

        return all_data


