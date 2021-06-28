package knn

import "knn_distribuido/utils"

func MergeSort(arr_tuples []utils.Tuple, ordered_tuples chan []utils.Tuple) {
	if len(arr_tuples) > 1 {
		ch_left := make(chan []utils.Tuple)
		ch_right := make(chan []utils.Tuple)
		mid := len(arr_tuples) / 2

		go MergeSort(arr_tuples[:mid], ch_left)
		go MergeSort(arr_tuples[mid:], ch_right)
		res_left := <-ch_left
		res_right := <-ch_right

		ordered_tuples <- Merge(res_left, res_right)
		return
	} else {
		ordered_tuples <- arr_tuples
		return
	}
}

func Merge(arr_left []utils.Tuple, arr_right []utils.Tuple) (merged []utils.Tuple) {
	merged = make([]utils.Tuple, len(arr_left)+len(arr_right))
	i := 0
	j := 0

	for k := 0; k < cap(merged); k++ {
		if i >= len(arr_left) {
			merged[k] = arr_right[j]
			j++
		} else if j >= len(arr_right) {
			merged[k] = arr_left[i]
			i++
		} else if arr_left[i].Value < arr_right[j].Value {
			merged[k] = arr_left[i]
			i++
		} else {
			merged[k] = arr_right[j]
			j++
		}
	}

	return merged
}
