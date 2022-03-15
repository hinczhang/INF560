#ifndef QUEUE_H
#define QUEUE_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "nbody.cuh"
#define ElementType node_t* //存储数据元素的类型
#define MAXSIZE 5000

typedef struct {
    ElementType data[MAXSIZE];
    int front; //记录队列头元素位置
    int rear; //记录队列尾元素位置
    int size; //存储数据元素的个数
}Queue;

__device__ Queue* CreateQueue();
__device__ int IsFullQ(Queue* q);
__device__ void AddQ(Queue* q, ElementType item);
__device__ int IsEmptyQ(Queue* q);
__device__ ElementType DeleteQ(Queue* q);

#endif