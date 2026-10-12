package tensai

import "testing"

func TestEnsureMatrixReshapesWithinCapacity(t *testing.T) {
	m := NewMatrix(8, 4)
	data := &m.Data[0]

	// Same shape: the same matrix.
	if got := EnsureMatrix(m, 8, 4); got != m {
		t.Fatal("same shape did not return m")
	}
	// Fewer rows: m reshaped in place over the same backing array.
	got := EnsureMatrix(m, 5, 4)
	if got != m || got.Rows != 5 || got.Cols != 4 || len(got.Data) != 20 || &got.Data[0] != data {
		t.Fatalf("shrink = %dx%d len %d (same matrix %v, same array %v), want m reshaped to 5x4",
			got.Rows, got.Cols, len(got.Data), got == m, &got.Data[0] == data)
	}
	// Back to more rows within the original capacity: still the same array.
	got = EnsureMatrix(m, 2, 16)
	if got != m || len(got.Data) != 32 || &got.Data[0] != data {
		t.Fatal("regrow within capacity allocated")
	}
	// Past the capacity: a new, zeroed matrix; m is left as it was.
	got = EnsureMatrix(m, 9, 4)
	if got == m || got.Rows != 9 || got.Cols != 4 || len(got.Data) != 36 {
		t.Fatalf("grow past capacity = %dx%d len %d (same matrix %v), want a new 9x4", got.Rows, got.Cols, len(got.Data), got == m)
	}
	for i, v := range got.Data {
		if v != 0 {
			t.Fatalf("new matrix element %d = %v, want 0", i, v)
		}
	}
	if EnsureMatrix(nil, 3, 3) == nil {
		t.Fatal("nil m did not allocate")
	}
}
