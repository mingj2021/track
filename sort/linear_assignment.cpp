#include "linear_assignment.h"
#include <iostream>
#include <math.h>


std::vector<std::vector<int>> linear_assignment(std::vector<std::vector<float>> &cost_matrix)
{
	Hungarian hungarian(cost_matrix);
	
	return hungarian.runMunkres();;
}




HungarianState::HungarianState(std::vector<std::vector<float>> &cost_matrix)
{

	/*
	if there are more rows(n) than columns(m), then the algorithm
	will not be able to work correctly.Therefore, we transpose the cost
	function when needed, just have to remember to swap the result columns
	back later.
	*/

	transposed = (cost_matrix[0].size() < cost_matrix.size());
	if (transposed)
	{
		for (int c = 0; c < cost_matrix[0].size(); c++)
		{
			std::vector<float> vec;
			for (int r = 0; r < cost_matrix.size(); r++)
			{
				vec.push_back(cost_matrix[r][c]);
			}
			C.emplace_back(vec);
		}
	}
	else
	{
		C = cost_matrix;
	}
	
	//at this point, m >= n
	nrow = C.size();
	ncol = C[0].size();
	C_org = C;
	RowCover = std::vector<int>(nrow, 0);
	ColCover = std::vector<int>(ncol, 0);

	path_row_0 = 0;
	path_col_0 = 0;

	for (int r = 0; r < nrow + ncol + 1; r++)
	{
		path.emplace_back(std::vector<int>(2, 0));
	}

	for (int r = 0; r < nrow; ++r)
	{
		M.emplace_back(std::vector<int>(ncol, 0));
	}
}

HungarianState::~HungarianState()
{

}

void HungarianState::resetMaskandCovers()
{
	for (int r = 0; r < nrow; ++r)
	{
		RowCover[r] = 0;
		for (int c = 0; c < ncol; ++c)
		{
			M[r][c] = 0;
		}
	}

	for (int c = 0; c < ncol; ++c)
		ColCover[c] = 0;
	return;
}

void HungarianState::showCostMatrix()
{
	using std::cout;
	using std::endl;
	for (int r = 0; r < nrow; r++)
	{
		for (int c = 0; c < ncol; c++)
			cout << C[r][c] << "  ";
		cout << endl;
	}
}

void HungarianState::showMaskMatrix()
{
	using std::cout;
	using std::endl;
	cout << endl << "col covered" << endl;
	for (int c = 0; c < ncol; c++)
		cout << ColCover[c] << "  ";

	cout << endl << "row covered" << endl;
	for (int r = 0; r < nrow; r++)
		cout << RowCover[r] << "  ";

	cout << endl << "mask matrix" << endl;
	for (int r = 0; r < nrow; r++)
	{
		for (int c = 0; c < ncol; c++)
			cout << M[r][c] << "  ";
		cout << endl;
	}

}

std::vector<std::vector<int>> HungarianState::getResults()
{
	std::vector<std::vector<int>> rults;
	std::vector<int> matched_row;
	std::vector<int> unmatched_row;

	std::vector<int> matched_col;
	std::vector<int> unmatched_col;

	for (int r = 0; r < nrow; r++)
	{
		for (int c = 0; c < ncol; c++)
			if (M[r][c] == 1)
			{
				if (1 - C_org[r][c] < 0.3)
				{
					unmatched_row.push_back(r);
					unmatched_col.push_back(c);
				}
				else
				{
					matched_row.push_back(r);
					matched_col.push_back(c);
				}
			}
	}

	for (int r = 0; r < nrow; r++)
	{
		if (find(matched_row.begin(), matched_row.end(), r) == matched_row.end())
		{
			unmatched_row.push_back(r);
		}
	}

	for (int c = 0; c < ncol; c++)
	{
		if (find(matched_col.begin(), matched_col.end(), c) == matched_col.end())
		{
			unmatched_col.push_back(c);
		}
	}
	if (transposed)
	{
		rults.emplace_back(matched_col);
		rults.emplace_back(matched_row);

		rults.emplace_back(unmatched_col);
		rults.emplace_back(unmatched_row);
	}
	else
	{
		rults.emplace_back(matched_row);
		rults.emplace_back(matched_col);

		rults.emplace_back(unmatched_row);
		rults.emplace_back(unmatched_col);
	}

	return rults;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////
Hungarian::Hungarian(std::vector<std::vector<float>> &cost_matrix):m_state(cost_matrix)
{

}
Hungarian::~Hungarian()
{

}


//For each row of the cost matrix, find the smallest element and subtract
//it from every element in its row. When finished, Go to step 2.
void Hungarian::step_one()
{
	float min_in_row;
	for (int r = 0; r < m_state.nrow; ++r)
	{
		min_in_row = m_state.C[r][0];
		for (int c = 0; c < m_state.ncol; c++)
			if (m_state.C[r][c] < min_in_row)
			{
				min_in_row = m_state.C[r][c];
			}
		for (int c = 0; c < m_state.ncol; c++)
			m_state.C[r][c] -= min_in_row;
	}
	step = 2;
	return;
}


//Find a zero(Z) in the resulting matrix. if there is no starred
//zero in its row or column. star Z. Repeat for each element in the
//matrix. go to step 3.
void Hungarian::step_two()
{
	for (int r = 0; r < m_state.nrow; r++)
		for (int c = 0; c < m_state.ncol; c++)
		{
			if (m_state.C[r][c] <= 1e-6 && m_state.RowCover[r] == 0 && m_state.ColCover[c] == 0)
			{
				m_state.M[r][c] = 1;
				m_state.RowCover[r] = 1;
				m_state.ColCover[c] = 1;
			}
		}
	for (int r = 0; r < m_state.nrow; r++)
		m_state.RowCover[r] = 0;
	for (int c = 0; c < m_state.ncol; c++)
		m_state.ColCover[c] = 0;
	step = 3;
	return;
}


//Cover each column containing a starred zero. if K columns are covered,
//the starred zeros describe a complete set of unique assignments. in
//this case, go to done, otherwise, go to step 4.
void  Hungarian::step_three()
{
	int colcount;
	for (int r = 0; r < m_state.nrow; r++)
		for (int c = 0; c < m_state.ncol; c++)
			if (m_state.M[r][c] == 1)
			{
				m_state.ColCover[c] = 1;
			}
	colcount = 0;
	for (int c = 0; c < m_state.ncol; c++)
		if (m_state.ColCover[c] == 1)
			colcount += 1;
	if (colcount >= m_state.ncol || colcount >= m_state.nrow)
		step = 7;
	else
		step = 4;
	return;
}

//methods to support step 4
void Hungarian::find_a_zero(int& row, int& col)
{
	int r = 0;
	int c;
	bool done;
	row = -1;
	col = -1;
	done = false;
	while (!done)
	{
		c = 0;
		while (true)
		{
			if (m_state.C[r][c] <= 1e-6 && m_state.RowCover[r] == 0 && m_state.ColCover[c] == 0)
			{
				row = r;
				col = c;
				done = true;
			}
			c += 1;
			if (c >= m_state.ncol || done)
				break;
		}
		r += 1;
		if (r >= m_state.nrow)
			done = true;
	}
	return;
}

//methods to support step 4
bool Hungarian::star_in_row(int row)
{
	bool tmp = false;
	for (int c = 0; c < m_state.ncol; c++)
		if (m_state.M[row][c] == 1)
			tmp = true;
	return tmp;
}

void Hungarian::find_star_in_row(int row, int& col)
{
	col = -1;
	for (int c = 0; c < m_state.ncol; c++)
		if (m_state.M[row][c] == 1)
			col = c;
	return;
}



//Find a noncovered zero and prime it. if there is no starred zero
//in the row containing this primed zero go to step 5. otherwise,
//cover this row and uncover the column containing the starred zero,
//continue in this manner until there are no uncovered zeros left.
//Save the smallest uncovered value and go to step 6.
void Hungarian::step_four()
{
	int row = -1;
	int col = -1;
	bool done;

	done = false;
	while (!done)
	{
		find_a_zero(row, col);
		if (row == -1)
		{
			done = true;
			step = 6;
		}
		else
		{
			m_state.M[row][col] = 2;
			if (star_in_row(row))
			{
				find_star_in_row(row, col);
				m_state.RowCover[row] = 1;
				m_state.ColCover[col] = 0;
			}
			else
			{
				done = true;
				step = 5;
				m_state.path_row_0 = row;
				m_state.path_col_0 = col;
			}
		}
	}
	return;
}


//methods to support step 5
void Hungarian::find_star_in_col(int c, int& r)
{
	r = -1;
	for (int i = 0; i < m_state.nrow; i++)
	{
		if (m_state.M[i][c] == 1)
			r = i;
	}
	return;
}


void Hungarian::find_prime_in_row(int r, int& c)
{
	for (int j = 0; j < m_state.ncol; j++)
		if (m_state.M[r][j] == 2)
			c = j;
	return;
}


void Hungarian::augment_path()
{
	for (int p = 0; p < m_state.path_count; p++)
		if (m_state.M[m_state.path[p][0]][m_state.path[p][1]] == 1)
			m_state.M[m_state.path[p][0]][m_state.path[p][1]] = 0;
		else
		{
			m_state.M[m_state.path[p][0]][m_state.path[p][1]] = 1;
		}
	return;
}


void Hungarian::clear_covers()
{
	for (int r = 0; r < m_state.nrow; r++)
		m_state.RowCover[r] = 0;
	for (int c = 0; c < m_state.ncol; c++)
		m_state.ColCover[c] = 0;
}


void Hungarian::erase_primes()
{
	for (int r = 0; r < m_state.nrow; r++)
		for (int c = 0; c < m_state.ncol; c++)
			if (m_state.M[r][c] == 2)
				m_state.M[r][c] = 0;
	return;
}



//Construct a series of alternating primed and starred zeros as follows,
//Let Z0 represent the uncovered primed zero found in step 4. let Z1 denote
//the starred zero in the column of Z0(if any), Let Z2 denote the primed zero
//in the row of Z1(there will always be one). continue until the series
//terminates at a primed zero that has no starred zero in its column,
//Unstar each starred zero of the series, star each primed zero of the series,
//erase all primes and uncover every line in the matrix. return to step 3.
void Hungarian::step_five()
{
	bool done;
	int r = -1;
	int c = -1;

	m_state.path_count = 1;
	m_state.path[m_state.path_count - 1][0] = m_state.path_row_0;
	m_state.path[m_state.path_count - 1][1] = m_state.path_col_0;
	done = false;
	while (!done)
	{
		find_star_in_col(m_state.path[m_state.path_count - 1][1], r);
		if (r > -1)
		{
			m_state.path_count += 1;
			m_state.path[m_state.path_count - 1][0] = r;
			m_state.path[m_state.path_count - 1][1] = m_state.path[m_state.path_count - 2][1];
		}
		else
		{
			done = true;
		}
		if (!done)
		{
			find_prime_in_row(m_state.path[m_state.path_count - 1][0], c);
			m_state.path_count += 1;
			m_state.path[m_state.path_count - 1][0] = m_state.path[m_state.path_count - 2][0];
			m_state.path[m_state.path_count - 1][1] = c;
		}
	}
	augment_path();
	clear_covers();
	erase_primes();
	step = 3;
}

void Hungarian::find_smallest(float& minval)
{
	for (int r = 0; r < m_state.nrow; r++)
		for (int c = 0; c < m_state.ncol; c++)
			if (m_state.RowCover[r] == 0 && m_state.ColCover[c] == 0)
				if (minval > m_state.C[r][c])
					minval = m_state.C[r][c];
	return;
}

void Hungarian::step_six()
{
	float minval = INT32_MAX;
	find_smallest(minval);
	for (int r = 0; r < m_state.nrow; r++)
		for (int c = 0; c < m_state.ncol; c++)
		{
			if (m_state.RowCover[r] == 1)
				m_state.C[r][c] += minval;
			if (m_state.ColCover[c] == 0)
				m_state.C[r][c] -= minval;
		}
	step = 4;
}


std::vector<std::vector<int>> Hungarian::runMunkres()
{
	bool done = false;
	while (!done)
	{
		switch (step)
		{
		case 1:
			step_one();
			break;
		case 2:
			step_two();
			break;
		case 3:
			step_three();
			break;
		case 4:
			step_four();
			break;
		case 5:
			step_five();
			break;
		case 6:
			step_six();
			break;
		case 7:
			done = true;
			break;
		default:
			break;
		}
	}
	return m_state.getResults();
}