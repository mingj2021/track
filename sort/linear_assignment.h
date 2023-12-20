#pragma once
#include <vector>

std::vector<std::vector<int>> linear_assignment(std::vector<std::vector<float>> &cost_matrix);




class HungarianState
{
	/*State of one execution of the Hungarian algorithm.

	Parameters
	-----------------------------
	cost_matrix: 2D matrix
	The cost matrix.Does not have to be square.
	*/
public:
	HungarianState(std::vector<std::vector<float>> &cost_matrix);
	~HungarianState();

public:
	void resetMaskandCovers();
	void showCostMatrix();
	void showMaskMatrix();
	std::vector<std::vector<int>> getResults();
	

public:
	bool transposed = false;
	// ncol > = nrow
	int nrow, ncol;

	std::vector<std::vector<float>> C;
	std::vector<std::vector<float>> C_org;
	std::vector<std::vector<int>> M;
	std::vector<std::vector<int>> path;
	std::vector<int> RowCover;
	std::vector<int> ColCover;
	int path_row_0;
	int path_col_0;
	int path_count = 0;
	
	
};



class Hungarian
{
public:
	Hungarian(std::vector<std::vector<float>> &cost_matrix);
	~Hungarian();

public:
	std::vector<std::vector<int>> runMunkres();

private:
	void step_one();
	void step_two();
	void step_three();

	void find_a_zero(int& row, int& col);
	bool star_in_row(int row);
	void find_star_in_row(int row, int& col);
	void step_four();

	//methods to support step 5
	void find_star_in_col(int c, int& r);
	void find_prime_in_row(int r, int& c);
	void augment_path();
	void clear_covers();
	void erase_primes();
	void step_five();

	void find_smallest(float& minval);
	void step_six();

private:
	HungarianState m_state;
	int step = 1;
};