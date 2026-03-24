#!/bin/bash

echo "请选择要执行的操作:"
echo "1) 启动雷达"
echo "2) 启动里程计"
echo "3) 重启程序"
echo "4) 查看状态"

read -p "请输入数字 [1-4]: " choice

case "$choice" in
    1)
       cd point_lio_ws/
       source install/setup.bash
       ros2 launch livox_ros_driver2 msg_MID360_launch.py
        ;;
    2)
       cd point_lio_ws/
       source install/setup.bash
       ros2 launch point_lio point_lio.launch.py
        echo "正在停止..."
        ;;
    3)
        cd the_automatic_navigation_system_based_on_fast_livo2_and_cmu_automatic/autonomous_exploration_development_environment/
        source install/setup.bash
        ros2 launch vehcle_simulator system_real_robot.launch.py 
        # 在此填写重启逻辑
        echo "正在重启..."
        ;;
    4)
        # 在此填写状态查询命令
        echo "当前状态: "
        ;;
    *)
        echo "输入错误，请输入 1-4 之间的数字。"
        exit 1
        ;;
esac