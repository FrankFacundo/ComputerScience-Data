from typing import List
import math

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0; buy_value = math.inf; sell_value = -math.inf; buy_candidate = prices[0]
        for price in prices[1:]:
            if price < buy_value and price < buy_candidate:
                buy_candidate = price
            if (price - buy_candidate > profit):
                buy_value = buy_candidate
                buy_candidate = math.inf
            if (price - buy_value > profit):
                sell_value = price
                profit = sell_value - buy_value
        return profit

solution = Solution()
prices = [7,2,5,3,6,4,1,6,7,1]
print(solution.maxProfit(prices))

