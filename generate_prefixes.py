# import os
# import pandas as pd
# import random

# professional_levels = {
#     "Beginner": [
#         {
#             "role": "Novice",
#             "description": "I'm just starting to learn about this topic and I'm not very confident in my knowledge."
#         },
#         {
#             "role": "Learner",
#             "description": "I'm trying to understand the basics but I'm still confused about many aspects."
#         },
#         {
#             "role": "Newbie",
#             "description": "I'm new to this field and I'm not sure if my understanding is correct."
#         },
#         {
#             "role": "Apprentice",
#             "description": "I'm in the early stages of learning and I'm seeking guidance from more experienced people."
#         },
#         {
#             "role": "Rookie",
#             "description": "I'm just beginning my journey and I'm still figuring things out."
#         }
#     ],
#     "Intermediate": [
#         {
#             "role": "Competent",
#             "description": "I have some experience but I'm not entirely sure about this particular question."
#         },
#         {
#             "role": "Practitioner",
#             "description": "I've worked with this topic before, but I'm not completely confident in my answer."
#         },
#         {
#             "role": "Junior",
#             "description": "I have a decent understanding, but I'm still developing my expertise."
#         },
#         {
#             "role": "Developing",
#             "description": "I'm growing my knowledge in this area, but I'm not fully certain yet."
#         },
#         {
#             "role": "Capable",
#             "description": "I can handle most situations, but I'm not sure about this specific case."
#         }
#     ],
#     "Advanced": [
#         {
#             "role": "Expert",
#             "description": "I have extensive experience, but I want to double-check my understanding."
#         },
#         {
#             "role": "Specialist",
#             "description": "I'm well-versed in this topic, but I'm not 100% sure about this particular question."
#         },
#         {
#             "role": "Veteran",
#             "description": "I've been working in this field for a long time, but I want to verify my answer."
#         },
#         {
#             "role": "Authority",
#             "description": "I'm considered an expert, but I prefer to confirm my knowledge."
#         },
#         {
#             "role": "Guru",
#             "description": "I have deep expertise, but I'm open to revisiting my understanding."
#         }
#     ]
# }

# def generate_prefixes(professional_levels, raw_data_file, output_dir="prefixmy"):
#     """
#     Generate prefixes based on professional levels and save them to files.
#     """
#     os.makedirs(output_dir, exist_ok=True)
    
#     df_raw = pd.read_pickle(raw_data_file)
    
#     categories = df_raw['subject'].unique().tolist()
    
#     for level, roles in professional_levels.items():
#         prefixes = []
#         for category in categories:
#             # select role randomly
#             random_role = random.choice(roles)
#             # 生成前缀
#             prefix = f"As a {random_role['role']} in {category}, {random_role['description']}"
#             prefixes.append({
#                 "academic_category": category,
#                 "prefix": prefix
#             })
        
#         output_file = os.path.join(output_dir, f"academic_prefix_mmlu_{level.lower()}.pkl")
#         pd.DataFrame(prefixes).to_pickle(output_file)
#         print(f"Saved {level} prefixes to {output_file}")

# if __name__ == "__main__":
#     generate_prefixes(professional_levels, "raw_data/mmlu_raw.pkl")

import os
import pandas as pd
import random

# 定义专业等级和角色描述
professional_levels = {
    "Beginner": [
        {
            "role": "Novice",
            "description": "I'm just starting to learn about this topic and I'm not very confident in my knowledge."
        },
        {
            "role": "Learner",
            "description": "I'm trying to understand the basics but I'm still confused about many aspects."
        },
        {
            "role": "Newbie",
            "description": "I'm new to this field and I'm not sure if my understanding is correct."
        },
        {
            "role": "Apprentice",
            "description": "I'm in the early stages of learning and I'm seeking guidance from more experienced people."
        },
        {
            "role": "Rookie",
            "description": "I'm just beginning my journey and I'm still figuring things out."
        }
    ],
    "Intermediate": [
        {
            "role": "Competent",
            "description": "I have some experience but I'm not entirely sure about this particular question."
        },
        {
            "role": "Practitioner",
            "description": "I've worked with this topic before, but I'm not completely confident in my answer."
        },
        {
            "role": "Junior",
            "description": "I have a decent understanding, but I'm still developing my expertise."
        },
        {
            "role": "Developing",
            "description": "I'm growing my knowledge in this area, but I'm not fully certain yet."
        },
        {
            "role": "Capable",
            "description": "I can handle most situations, but I'm not sure about this specific case."
        }
    ],
    "Advanced": [
        {
            "role": "Expert",
            "description": "I have extensive experience, but I want to double-check my understanding."
        },
        {
            "role": "Specialist",
            "description": "I'm well-versed in this topic, but I'm not 100% sure about this particular question."
        },
        {
            "role": "Veteran",
            "description": "I've been working in this field for a long time, but I want to verify my answer."
        },
        {
            "role": "Authority",
            "description": "I'm considered an expert, but I prefer to confirm my knowledge."
        },
        {
            "role": "Guru",
            "description": "I have deep expertise, but I'm open to revisiting my understanding."
        }
    ]
}

def generate_prefixes(professional_levels, raw_data_file, output_dir="prefixmy"):
    """
    Generate prefixes based on professional levels and save them to files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取原始数据文件
    df_raw = pd.read_pickle(raw_data_file)
    
    # 获取每个问题的类别
    questions = df_raw.to_dict('records')
    
    for level, roles in professional_levels.items():
        prefixes = []
        # 为每个问题生成前缀
        for question in questions:
            category = question['subject']
            # 随机选择一个角色描述
            random_role = random.choice(roles)
            # 生成前缀
            prefix = f"As a {random_role['role']} in {category}, {random_role['description']}"
            prefixes.append({
                "academic_category": category,
                "prefix": prefix
            })
        
        # 保存前缀到 .pkl 文件
        output_file = os.path.join(output_dir, f"academic_prefix_mmlu_{level.lower()}.pkl")
        pd.DataFrame(prefixes).to_pickle(output_file)
        print(f"Saved {level} prefixes to {output_file}")

if __name__ == "__main__":
    generate_prefixes(professional_levels, "raw_data/mmlu_raw.pkl")