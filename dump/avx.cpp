//transpose 4 vectors of 16 floats
void avx::transpose16x4(std::span<V16f> data) {
	V16f tmp[8];

	tmp[0] = _mm512_unpacklo_ps(data[0], data[1]);
	tmp[1] = _mm512_unpackhi_ps(data[0], data[1]);
	tmp[2] = _mm512_unpacklo_ps(data[2], data[3]);
	tmp[3] = _mm512_unpackhi_ps(data[2], data[3]);

	data[0] = _mm512_shuffle_ps(tmp[0], tmp[2], 0b0100'0100);
	data[1] = _mm512_shuffle_ps(tmp[0], tmp[2], 0b1110'1110);
	data[2] = _mm512_shuffle_ps(tmp[1], tmp[3], 0b0100'0100);
	data[3] = _mm512_shuffle_ps(tmp[1], tmp[3], 0b1110'1110);
}


//transpose 8 vectors of 16 floats
//result is returned again in 8 vectors of 16 floats
//each vector contains two 'blocks' of 8 floats representing data rows
void avx::transpose16x8(std::span<V16f> data) {
	V16f tmp[8];

	tmp[0] = _mm512_unpacklo_ps(data[0], data[1]);
	tmp[1] = _mm512_unpackhi_ps(data[0], data[1]);
	tmp[2] = _mm512_unpacklo_ps(data[2], data[3]);
	tmp[3] = _mm512_unpackhi_ps(data[2], data[3]);
	tmp[4] = _mm512_unpacklo_ps(data[4], data[5]);
	tmp[5] = _mm512_unpackhi_ps(data[4], data[5]);
	tmp[6] = _mm512_unpacklo_ps(data[6], data[7]);
	tmp[7] = _mm512_unpackhi_ps(data[6], data[7]);

	data[0] = _mm512_shuffle_ps(tmp[0], tmp[2], 0b0100'0100);
	data[1] = _mm512_shuffle_ps(tmp[0], tmp[2], 0b1110'1110);
	data[2] = _mm512_shuffle_ps(tmp[1], tmp[3], 0b0100'0100);
	data[3] = _mm512_shuffle_ps(tmp[1], tmp[3], 0b1110'1110);
	data[4] = _mm512_shuffle_ps(tmp[4], tmp[6], 0b0100'0100);
	data[5] = _mm512_shuffle_ps(tmp[4], tmp[6], 0b1110'1110);
	data[6] = _mm512_shuffle_ps(tmp[5], tmp[7], 0b0100'0100);
	data[7] = _mm512_shuffle_ps(tmp[5], tmp[7], 0b1110'1110);

	tmp[0] = _mm512_shuffle_f32x4(data[0], data[4], 0b1000'1000);
	tmp[1] = _mm512_shuffle_f32x4(data[1], data[5], 0b1000'1000);
	tmp[2] = _mm512_shuffle_f32x4(data[2], data[6], 0b1000'1000);
	tmp[3] = _mm512_shuffle_f32x4(data[3], data[7], 0b1000'1000);
	tmp[4] = _mm512_shuffle_f32x4(data[0], data[4], 0b1101'1101);
	tmp[5] = _mm512_shuffle_f32x4(data[1], data[5], 0b1101'1101);
	tmp[6] = _mm512_shuffle_f32x4(data[2], data[6], 0b1101'1101);
	tmp[7] = _mm512_shuffle_f32x4(data[3], data[7], 0b1101'1101);

	data[0] = _mm512_shuffle_f32x4(tmp[0], tmp[1], 0b1000'1000);
	data[1] = _mm512_shuffle_f32x4(tmp[2], tmp[3], 0b1000'1000);
	data[2] = _mm512_shuffle_f32x4(tmp[4], tmp[5], 0b1000'1000);
	data[3] = _mm512_shuffle_f32x4(tmp[6], tmp[7], 0b1000'1000);
	data[4] = _mm512_shuffle_f32x4(tmp[0], tmp[1], 0b1101'1101);
	data[5] = _mm512_shuffle_f32x4(tmp[2], tmp[3], 0b1101'1101);
	data[6] = _mm512_shuffle_f32x4(tmp[4], tmp[5], 0b1101'1101);
	data[7] = _mm512_shuffle_f32x4(tmp[6], tmp[7], 0b1101'1101);
}

