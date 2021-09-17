#ifndef time_m_hpp_included_
#define time_m_hpp_included_

//
// Copyright(c) Daniel Knüttel 2020
//

//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.


#include <chrono>
#include <iostream>

/*
 * Use those macros around a BLOCK(!) of code to log its 
 * execution time in miliseconds. 
 * Note that this will add an extra block. Thus, variables declared between
 * CLOG_TIMEIT_START and CLOG_TIMEIT_END will be invisible to the rest of the
 * code.
 *
 * Example:
 *
 * int
 * cool_function(int n)
 * {
 *      for(int i = 0; i < n; i++)
 *      {
 *          CLOG_TIMEIT_START
 *          for(int j = 0; j < i; j++)
 *          {
 *              do_something_cool(i, j);
 *          }
 *          CLOG_TIMEIT_END("loop over do_something_cool")
 *      }
 * }
 * */

#define CLOG_TIMEIT_START {\
    auto clog_timeit_start = std::chrono::high_resolution_clock::now();
#define CLOG_TIMEIT_END(name) \
    auto clog_timeit_stop = std::chrono::high_resolution_clock::now(); \
    std::chrono::duration<double, std::milli> clog_timeit_elapsed = clog_timeit_stop - clog_timeit_start; \
    std::clog << "> " << name << " ELAPSED: " << clog_timeit_elapsed.count() << " ms\n"; \
    } \

#endif
