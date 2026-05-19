cd /home/zhanglingjun.zlj/code/Bench2Drive/resultfailed3/Scenarios && \
for d in failed2_*_qw_tj_*; do
  [ -d "$d" ] || continue
  new_name="${d#failed2_}"
  new_name="${new_name#*_qw_tj_}"

  if [ -e "$new_name" ]; then
    echo "跳过: $d -> $new_name (目标已存在)"
  else
    echo "重命名: $d -> $new_name"
    mv "$d" "$new_name"
  fi
done
