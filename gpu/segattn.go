package gpu

// RowFlag marks a packed row for Device.SegmentAttention; NewSegments
// takes one per row.
type RowFlag uint8

const (
	// RowQueryOnly keeps a row out of the keys: no query attends to it,
	// though its own query still attends to the keys of its segment.
	RowQueryOnly RowFlag = 1 << iota
	// RowGated puts a key in the gated group when SegAttn.Gate is set: a
	// query sees the ungated keys through one softmax and the gated ones
	// through another, weighted per head (see SegAttn.Gate). Without a
	// gate the flag is ignored.
	RowGated
	// RowSeesAll makes a row's query attend to every key of its segment
	// through one plain softmax, gated keys included, even when
	// SegAttn.Gate is set. Without a gate every query does that anyway.
	RowSeesAll
)

// SegAttn describes one Device.SegmentAttention call: multi-head
// scaled dot-product attention over variable-length segments packed into
// one run of rows with no padding -- many short sequences (the positions
// of a batch of games, say) in a single dispatch.
//
// Seg splits the packed rows into segments, and each query attends only
// to the keys of its own segment that are not RowQueryOnly. K and V hold
// one row per packed row; head h of a row is the dh = Dim/Heads columns
// starting at KCol+h*dh (VCol+h*dh). Q holds one row per query, head h at
// QCol+h*dh. Q, K and V may be one tensor -- the fused output of a q|k|v
// projection with KCol and VCol set to Dim and 2*Dim -- and their rows may
// be wider than Dim; Dim, dh, the columns and the row widths must be
// multiples of four.
//
// Without QRows there is one query per packed row, in order. QRows (an
// UploadIndices tensor) lists instead the packed row of each query, so a
// last layer that reads only a few rows of every segment computes just
// those: query i then has its Q row i, the segment and flags of packed
// row QRows[i], and output row i.
//
// Gate, when set, holds one weight per head (at most 32 heads) and splits
// the keys of every segment into the RowGated ones and the rest. A query
// whose row is not RowSeesAll gets softmax-attention over the ungated keys
// plus Gate[h] times softmax-attention over the gated keys (a learned
// gate alpha passes tanh(alpha)); a RowSeesAll query attends to all keys
// as usual. A Gate weight of 0 therefore gives a query the result it
// would have without the gated keys at all.
//
// A query with no key to attend to -- in a group, under a gate -- gets
// zeros from it, and a query whose packed row lies outside every segment
// gets zeros outright.
type SegAttn struct {
	Seg   *Segments
	Heads int
	Dim   int // heads*dh; 0 takes Q's row width

	Q, K, V          *Tensor
	QCol, KCol, VCol int

	QRows *Tensor   // packed row of each query; nil = every packed row
	Gate  []float32 // per-head weight of the gated keys; nil = no gate
	Scale float32   // multiplies q.k; 0 = 1/sqrt(dh)
}
