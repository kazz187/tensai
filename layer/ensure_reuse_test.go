package layer

import (
	"testing"

	"github.com/mattn/tensai"
)

// TestForwardReusesBuffersAcrossRowCounts checks that LayerNorm and Softmax in training mode (scratch reuse) do not
// allocate their outputs again when the row count changes between calls but stays within what they already hold —
// an encoder over sequences of a different length each call — and that the results match fresh layers.
func TestForwardReusesBuffersAcrossRowCounts(t *testing.T) {
	const cols = 16
	input := func(rows int) *tensai.Matrix {
		m := tensai.NewMatrix(rows, cols)
		for i := range m.Data {
			m.Data[i] = tensai.Float(i%7) - 3.25
		}
		return m
	}
	ln := NewLayerNorm()
	if _, err := ln.Init(cols, nil); err != nil {
		t.Fatal(err)
	}
	ln.SetTraining(true)
	sm := &Softmax{}
	sm.SetTraining(true)
	if _, err := ln.Forward(input(40)); err != nil { // the largest first
		t.Fatal(err)
	}
	if _, err := sm.Forward(input(40)); err != nil {
		t.Fatal(err)
	}
	for _, rows := range []int{17, 33, 40, 5} {
		in := input(rows)
		allocs := testing.AllocsPerRun(10, func() {
			if _, err := ln.Forward(in); err != nil {
				t.Fatal(err)
			}
			if _, err := sm.Forward(in); err != nil {
				t.Fatal(err)
			}
		})
		if allocs != 0 {
			t.Fatalf("rows %d: %v allocations per Forward pair, want 0", rows, allocs)
		}
		gotLN, _ := ln.Forward(in)
		gotSM, _ := sm.Forward(in)
		fresh := NewLayerNorm()
		if _, err := fresh.Init(cols, nil); err != nil {
			t.Fatal(err)
		}
		wantLN, _ := fresh.Forward(in)
		wantSM, _ := (&Softmax{}).Forward(in)
		for i := range wantLN.Data {
			if gotLN.Data[i] != wantLN.Data[i] || gotSM.Data[i] != wantSM.Data[i] {
				t.Fatalf("rows %d element %d: reused (%v, %v) != fresh (%v, %v)", rows, i, gotLN.Data[i], gotSM.Data[i], wantLN.Data[i], wantSM.Data[i])
			}
		}
	}
}
