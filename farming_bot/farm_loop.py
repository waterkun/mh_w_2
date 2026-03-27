"""自动刷怪循环 — 接任务 → 准备 → 找怪 → 战斗 → 放弃任务 → 重复.

使用方式:
  python -m farming_bot.farm_loop

按 Ctrl+C 停止.
"""

import time
import winsound

from farming_bot.start_mission.start_mission import StartMission, beep
from farming_bot.prepare_mission.prepare_mission import PrepareMission
from farming_bot.find_monster.find_monster import FindMonster
from farming_bot.attack_monster.attack_monster import AttackMonster
from farming_bot.abort_mission.abort_mission import AbortMission


def main():
    print("=" * 50)
    print("MH Wilds 自动循环 (接任务 → 准备 → 找怪 → 战斗 → 放弃)")
    print("按 Ctrl+C 停止")
    print("=" * 50)
    print()

    # 倒计时
    for i in range(10, 0, -1):
        print(f"  {i} 秒后开始... 请切到游戏窗口")
        beep(800, 100)
        time.sleep(1)

    beep(1200, 500)
    print()

    # starter 创建 pad/sct/monitor, 其他模块复用
    starter = StartMission()
    pad = starter._pad
    sct = starter._sct
    monitor = starter._monitor

    finder = FindMonster(pad=pad, sct=sct, monitor=monitor)
    finder.start_display()

    # display callback: 各模块调用来更新窗口标签 (不阻塞, 只改一个字符串)
    display_cb = finder.set_display_label

    starter._display_callback = display_cb
    preparer = PrepareMission(pad=pad, sct=sct, monitor=monitor, display_callback=display_cb)
    attacker = AttackMonster(pad=pad, sct=sct, monitor=monitor, display_thread=finder._display)
    aborter = AbortMission(pad=pad, sct=sct, monitor=monitor, display_callback=display_cb)

    cycle = 0

    try:
        while True:
            cycle += 1
            print(f"\n{'='*50}")
            print(f"循环 #{cycle}")
            print(f"{'='*50}")

            display_cb("START_MISSION")
            success = starter.run()
            if success:
                display_cb("PREPARE")
                preparer.prepare()

                # 找怪 ↔ 战斗 循环
                while True:
                    display_cb("FIND_MONSTER")
                    found = finder.find()
                    if not found:
                        print("找不到怪物, 放弃任务")
                        break

                    display_cb("ATTACK")
                    result = attacker.attack(timeout=180)

                    if result == "lost":
                        print("怪物丢失, 重新找怪...")
                        display_cb("FIND_MONSTER")
                        continue
                    else:
                        print(f"战斗结束: {result}")
                        break

                display_cb("ABORT_MISSION")
                abort_success = aborter.abort()
                if abort_success:
                    print("放弃成功, 继续下一轮...")
                else:
                    print("放弃失败, 等待 5 秒后重试...")
                    time.sleep(5.0)
            else:
                print("接任务失败, 等待 5 秒后重试...")
                time.sleep(5.0)

    except KeyboardInterrupt:
        print(f"\n\n停止! 共完成 {cycle} 轮循环")
        beep(600, 500)
    finally:
        finder.stop_display()


if __name__ == "__main__":
    main()
