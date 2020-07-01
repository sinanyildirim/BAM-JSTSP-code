for entry in results/consolidated/*/*
do
  julia icews_visualize.jl "$entry"
done
