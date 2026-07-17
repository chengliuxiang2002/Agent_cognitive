"""
数据库种子数据填充脚本

为认知记忆模块生成多样化测试数据，包含:
- 20个用户画像 (user_001 ~ user_020)
- 每个用户约15-20条交互记录
- 每个用户5-8个行为模式
- 多条反馈数据

使用方法: python seed_data.py
"""

import asyncio
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cognitive_memory.core.memory_manager import MemoryManager
from cognitive_memory.models.memory import (
    InteractionRecord,
    SceneContext,
    UserProfile,
    BehaviorPattern,
)

# ─── 数据池：20个用户，属性多样化 ──────────────────────

NAMES = [
    "张明", "李芳", "王建国", "陈晓", "刘洋", "赵丽", "孙磊", "周婷",
    "吴强", "郑雨", "钱伟", "马静", "黄涛", "林雪", "何军", "罗敏",
    "梁峰", "宋瑶", "唐杰", "韩冰",
]

AGE_GROUPS = ["18-25", "26-35", "36-50", "51-65"]
TEMP_PREFS = [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0]
DRIVE_MODES = ["eco", "comfort", "sport", "normal"]
MUSIC_GENRES = [
    ["pop", "light_music", "podcast"],
    ["rock", "electronic", "jazz"],
    ["classical", "news", "oldies"],
    ["hiphop", "r&b", "ballad"],
    ["folk", "country", "indie"],
    ["metal", "punk", "ambient"],
    ["kpop", "jpop", "lofi"],
    ["soul", "funk", "disco"],
]
LOCATIONS = [
    "中关村", "国贸", "五道口", "西二旗", "望京", "金融街", "朝外",
    "三里屯", "上地", "亦庄", "顺义", "通州", "丰台科技园", "亚运村",
    "大望路", "西直门", "东直门", "知春路", "回龙观", "天通苑",
]
LOCATION_TYPES = ["home", "office", "gym", "mall", "restaurant", "park", "hospital", "school"]
WEATHER = ["sunny", "cloudy", "rainy", "snowy", "windy", "foggy"]
ROAD_TYPES = ["highway", "urban", "suburban", "mountain", "tunnel"]
TIME_SLOTS = ["morning", "afternoon", "evening", "night"]
INTENTS = [
    "navigate", "play_music", "adjust_temperature", "switch_mode",
    "check_weather", "find_parking", "call_contact", "send_message",
    "set_reminder", "safety_alert", "adjust_seat", "adjust_mirror",
]
INTERACTION_TYPES = ["voice_command", "touch_input", "gesture", "automatic"]

DRIVING_HABITS = [
    ["平稳驾驶", "偏好高速", "避免拥堵"],
    ["激进驾驶", "喜欢山路", "大油门起步"],
    ["省油驾驶", "车速稳定", "预判减速"],
    ["快速变道", "跟车紧", "频繁超车"],
    ["缓慢起步", "保持车距", "礼让行人"],
]
INTERACTION_STYLES = ["minimal", "moderate", "verbose"]
VOICE_STYLES = ["concise", "detailed", "friendly", "professional"]


def make_user(idx: int) -> dict:
    return {
        "user_id": f"{idx:03d}",
        "name": NAMES[idx - 1],
        "age_group": random.choice(AGE_GROUPS),
        "temperature_preference": random.choice(TEMP_PREFS),
        "driving_mode_preference": random.choice(DRIVE_MODES),
        "music_preferences": random.choice(MUSIC_GENRES),
        "frequent_routes": [
            {"from": f"家({random.choice(LOCATIONS)})", "to": f"公司({random.choice(LOCATIONS)})", "frequency": random.randint(3, 6)},
            {"from": f"公司({random.choice(LOCATIONS)})", "to": f"家({random.choice(LOCATIONS)})", "frequency": random.randint(3, 6)},
            {"from": f"家({random.choice(LOCATIONS)})", "to": f"休闲({random.choice(LOCATIONS)})", "frequency": random.randint(1, 4)},
        ],
        "active_hours": [
            {"day": "weekday", "hours": sorted(random.sample(range(6, 23), random.randint(3, 6)))},
            {"day": "weekend", "hours": sorted(random.sample(range(8, 23), random.randint(3, 5)))},
        ],
        "driving_habits": random.choice(DRIVING_HABITS),
        "interaction_style": random.choice(INTERACTION_STYLES),
        "voice_style": random.choice(VOICE_STYLES),
    }


def make_interactions(user: dict) -> list[InteractionRecord]:
    now = datetime.now()
    uid = user["user_id"]
    records = []

    # 通勤导航 (4-6条)
    for i in range(random.randint(4, 6)):
        route = random.choice(user["frequent_routes"][:2])
        records.append(InteractionRecord(
            user_id=uid,
            interaction_type=random.choice(["voice_command", "touch_input"]),
            intent="navigate",
            raw_input=f"导航到{route['to'].split('(')[0]}",
            scene_context=SceneContext(
                time_of_day="morning" if i % 2 == 0 else "evening",
                location_type="home" if i % 2 == 0 else "office",
                weather=random.choice(WEATHER),
                traffic_condition=random.choice(["smooth", "congested", "moderate"]),
            ),
            system_response={"action": "navigate", "destination": route["to"]},
            response_time_ms=random.randint(180, 350),
            was_successful=random.random() > 0.05,
            timestamp=now - timedelta(days=random.randint(0, 30), hours=random.randint(0, 12)),
        ))

    # 温度调节 (2-3条)
    temp = user["temperature_preference"]
    for i in range(random.randint(2, 3)):
        records.append(InteractionRecord(
            user_id=uid,
            interaction_type=random.choice(["voice_command", "touch_input"]),
            intent="adjust_temperature",
            raw_input=f"温度调到{temp}度" if random.random() > 0.3 else f"有点{random.choice(['冷', '热'])}",
            scene_context=SceneContext(
                time_of_day=random.choice(TIME_SLOTS),
                weather=random.choice(["cold", "hot", "rainy", "snowy"]),
            ),
            system_response={"action": "set_temp", "value": temp},
            response_time_ms=random.randint(120, 250),
            was_successful=True,
            timestamp=now - timedelta(days=random.randint(0, 20), hours=random.randint(0, 10)),
        ))

    # 音乐 (2-3条)
    for i in range(random.randint(2, 3)):
        genre = random.choice(user["music_preferences"])
        records.append(InteractionRecord(
            user_id=uid,
            interaction_type="voice_command",
            intent="play_music",
            raw_input=random.choice([f"播放{genre}音乐", f"来点{genre}", f"我想听{genre}"]),
            scene_context=SceneContext(
                time_of_day=random.choice(["afternoon", "evening"]),
                location_type="road",
                weather=random.choice(WEATHER),
            ),
            system_response={"action": "play", "genre": genre},
            response_time_ms=random.randint(150, 300),
            was_successful=random.random() > 0.05,
            timestamp=now - timedelta(days=random.randint(0, 15), hours=random.randint(0, 8)),
        ))

    # 驾驶模式 (1-2条)
    mode = user["driving_mode_preference"]
    for i in range(random.randint(1, 2)):
        records.append(InteractionRecord(
            user_id=uid,
            interaction_type=random.choice(["touch_input", "voice_command"]),
            intent="switch_mode",
            raw_input=f"切换到{mode}模式",
            scene_context=SceneContext(
                time_of_day=random.choice(TIME_SLOTS),
                road_type=random.choice(ROAD_TYPES),
            ),
            system_response={"action": "switch_mode", "mode": mode},
            response_time_ms=random.randint(100, 200),
            was_successful=True,
            timestamp=now - timedelta(days=random.randint(0, 25), hours=random.randint(0, 6)),
        ))

    # 安全类 (1-2条)
    for i in range(random.randint(1, 2)):
        alert_type = random.choice(["碰撞预警", "车道偏离", "疲劳驾驶提醒", "超速警告", "盲区监测"])
        records.append(InteractionRecord(
            user_id=uid,
            interaction_type="automatic",
            intent="safety_alert",
            raw_input=alert_type,
            scene_context=SceneContext(
                time_of_day=random.choice(TIME_SLOTS),
                road_type=random.choice(["highway", "urban"]),
                traffic_condition=random.choice(["congested", "moderate"]),
            ),
            system_response={"action": "alert", "type": alert_type},
            response_time_ms=random.randint(30, 80),
            was_successful=True,
            timestamp=now - timedelta(days=random.randint(0, 30), hours=random.randint(0, 12)),
        ))

    # 其他类: 停车/天气/通话/座椅 (2-3条)
    other_intents = ["find_parking", "check_weather", "call_contact", "adjust_seat", "set_reminder"]
    for i in range(random.randint(2, 3)):
        intent = random.choice(other_intents)
        inputs = {
            "find_parking": "找附近停车场",
            "check_weather": "今天天气怎么样",
            "call_contact": "打电话给XX",
            "adjust_seat": "调整座椅位置",
            "set_reminder": "提醒我半小时后开会",
        }
        records.append(InteractionRecord(
            user_id=uid,
            interaction_type=random.choice(INTERACTION_TYPES),
            intent=intent,
            raw_input=inputs[intent],
            scene_context=SceneContext(
                time_of_day=random.choice(TIME_SLOTS),
                location_type=random.choice(LOCATION_TYPES),
                weather=random.choice(WEATHER),
            ),
            system_response={"action": intent, "acknowledged": True},
            response_time_ms=random.randint(150, 400),
            was_successful=random.random() > 0.1,
            timestamp=now - timedelta(days=random.randint(0, 20), hours=random.randint(0, 10)),
        ))

    return records


def make_profile(user: dict) -> UserProfile:
    return UserProfile(
        user_id=user["user_id"],
        name=user["name"],
        age_group=user["age_group"],
        temperature_preference=user["temperature_preference"],
        driving_mode_preference=user["driving_mode_preference"],
        music_preferences=user["music_preferences"],
        frequent_routes=user["frequent_routes"],
        active_hours=user["active_hours"],
        confidence_score=round(random.uniform(0.65, 0.95), 2),
        data_points_count=random.randint(15, 40),
        driving_habits=user["driving_habits"],
        interaction_style=user["interaction_style"],
        voice_assistant_style=user["voice_style"],
    )


def make_patterns(user: dict) -> list[BehaviorPattern]:
    now = datetime.now()
    uid = user["user_id"]
    routes = user["frequent_routes"]
    patterns = []

    # 通勤路线模式
    route_to = routes[0]["to"]
    route_from = routes[0]["from"]
    patterns.append(BehaviorPattern(
        user_id=uid,
        pattern_name=f"早晨通勤_{route_from.split('(')[0]}_{route_to.split('(')[0]}",
        pattern_type="route",
        trigger_conditions={"time_of_day": "morning", "location_type": "home"},
        expected_action={"navigate_to": route_to},
        occurrence_count=routes[0]["frequency"],
        confidence=round(random.uniform(0.80, 0.95), 2),
        first_observed=now - timedelta(days=random.randint(20, 40)),
        last_observed=now - timedelta(days=random.randint(0, 3)),
        related_context_keys=["morning", "home", "weekday", "commute"],
    ))

    patterns.append(BehaviorPattern(
        user_id=uid,
        pattern_name=f"傍晚归家_{route_to.split('(')[0]}_{route_from.split('(')[0]}",
        pattern_type="route",
        trigger_conditions={"time_of_day": "evening", "location_type": "office"},
        expected_action={"navigate_to": route_from},
        occurrence_count=routes[1]["frequency"],
        confidence=round(random.uniform(0.80, 0.95), 2),
        first_observed=now - timedelta(days=random.randint(20, 40)),
        last_observed=now - timedelta(days=random.randint(0, 3)),
        related_context_keys=["evening", "office", "weekday", "commute"],
    ))

    # 温度模式
    patterns.append(BehaviorPattern(
        user_id=uid,
        pattern_name=f"温度偏好_{user['temperature_preference']}°C",
        pattern_type="temperature",
        trigger_conditions={"time_of_day": random.choice(["morning", "evening"])},
        expected_action={"set_temperature": user["temperature_preference"]},
        occurrence_count=random.randint(8, 15),
        confidence=round(random.uniform(0.78, 0.92), 2),
        first_observed=now - timedelta(days=random.randint(15, 35)),
        last_observed=now - timedelta(days=random.randint(0, 5)),
        related_context_keys=["climate", "comfort", "morning"],
    ))

    # 音乐模式
    patterns.append(BehaviorPattern(
        user_id=uid,
        pattern_name=f"音乐偏好_{user['music_preferences'][0]}",
        pattern_type="media",
        trigger_conditions={"time_of_day": "afternoon"},
        expected_action={"preferred_genre": user["music_preferences"][0]},
        occurrence_count=random.randint(5, 12),
        confidence=round(random.uniform(0.75, 0.90), 2),
        first_observed=now - timedelta(days=random.randint(10, 25)),
        last_observed=now - timedelta(days=random.randint(0, 7)),
        related_context_keys=["afternoon", "music", "entertainment"],
    ))

    # 驾驶模式
    patterns.append(BehaviorPattern(
        user_id=uid,
        pattern_name=f"驾驶模式_{user['driving_mode_preference']}",
        pattern_type="driving_mode",
        trigger_conditions={"road_type": random.choice(["highway", "urban"])},
        expected_action={"switch_to": user["driving_mode_preference"]},
        occurrence_count=random.randint(4, 10),
        confidence=round(random.uniform(0.72, 0.88), 2),
        first_observed=now - timedelta(days=random.randint(10, 30)),
        last_observed=now - timedelta(days=random.randint(0, 8)),
        related_context_keys=["driving", "highway", "performance"],
    ))

    # 安全交互模式
    patterns.append(BehaviorPattern(
        user_id=uid,
        pattern_name="安全提醒确认",
        pattern_type="interaction",
        trigger_conditions={"intent": "safety_alert"},
        expected_action={"acknowledge_quickly": True},
        occurrence_count=random.randint(2, 6),
        confidence=round(random.uniform(0.65, 0.82), 2),
        first_observed=now - timedelta(days=random.randint(10, 25)),
        last_observed=now - timedelta(days=random.randint(0, 10)),
        related_context_keys=["safety", "alert", "highway"],
    ))

    # 第三个路线(休闲地点) - 部分用户有
    if len(routes) > 2 and random.random() > 0.3:
        third = routes[2]
        patterns.append(BehaviorPattern(
            user_id=uid,
            pattern_name=f"休闲路线_{third['to'].split('(')[0]}",
            pattern_type="route",
            trigger_conditions={"time_of_day": "weekend"},
            expected_action={"navigate_to": third["to"]},
            occurrence_count=third["frequency"],
            confidence=round(random.uniform(0.60, 0.78), 2),
            first_observed=now - timedelta(days=random.randint(15, 35)),
            last_observed=now - timedelta(days=random.randint(2, 14)),
            related_context_keys=["weekend", "leisure", "road"],
        ))

    return patterns


async def seed():
    db_path = str(Path(__file__).parent / "cognitive_memory.db")

    print(f"数据库路径: {db_path}")
    manager = MemoryManager(db_path=db_path)

    # 清除旧数据
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM user_feedback")
    conn.execute("DELETE FROM behavior_patterns")
    conn.execute("DELETE FROM interaction_records")
    conn.execute("DELETE FROM user_profiles")
    conn.execute("DELETE FROM long_term_memories")
    conn.commit()
    conn.close()
    print("旧数据已清除\n")

    total_interactions = 0
    total_patterns = 0

    for idx in range(1, 21):
        user = make_user(idx)
        uid = user["user_id"]
        name = user["name"]

        # 1. 画像
        profile = make_profile(user)
        await manager._profile_store.save_profile(profile)

        # 2. 交互
        interactions = make_interactions(user)
        for i in interactions:
            await manager._interaction_store.record(i)
        total_interactions += len(interactions)

        # 3. 行为模式
        patterns = make_patterns(user)
        for p in patterns:
            await manager._pattern_store.save_pattern(p)
        total_patterns += len(patterns)

        print(f"[{name}] ({uid}) 画像+{len(interactions)}条交互+{len(patterns)}个模式 | "
              f"偏好: {user['temperature_preference']}°C {user['driving_mode_preference']} "
              f"{user['music_preferences'][0]}")

    # 4. 反馈 (使用唯一ID避免与旧数据冲突)
    feedback_id_base = datetime.now().strftime("%Y%m%d%H%M%S")
    for i in range(20):
        uid = f"{random.randint(1, 20):03d}"
        ftype = random.choice(["like", "like", "like", "dislike"])  # 75% like
        await manager._feedback_store.record_feedback(
            feedback_id=f"fb_{feedback_id_base}_{i:04d}",
            user_id=uid,
            prediction_id=f"pred_{random.randint(0, 99):04d}",
            feedback_type=ftype,
            comment="" if ftype == "like" else random.choice(["不准确", "推荐不合理", "时机不对"]),
        )

    like_count = sum(1 for i in range(20) if random.choice([1, 1, 1, 0]) == 1)
    print(f"\n反馈数据: 20条\n")

    print("=" * 54)
    print("  种子数据填充完成!")
    print(f"  用户画像: {20} 个")
    print(f"  交互记录: {total_interactions} 条")
    print(f"  行为模式: {total_patterns} 个")
    print(f"  反馈数据: {20} 条")
    print("=" * 54)
    print("\n  可用用户ID: 001 ~ 020")
    print("  启动管理界面: uvicorn cognitive_memory.admin.server:app --port 8085")


if __name__ == "__main__":
    asyncio.run(seed())