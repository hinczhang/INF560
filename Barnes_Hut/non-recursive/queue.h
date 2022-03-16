#ifndef QUEUE_H
#define QUEUE_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "nbody.h"
#define ElementType node_t* //存储数据元素的类型
#define MAXSIZE 5000

typedef struct {
    ElementType data[MAXSIZE];
    int front; //记录队列头元素位置
    int rear; //记录队列尾元素位置
    int size; //存储数据元素的个数
}Queue;

Queue* CreateQueue();
int IsFullQ(Queue* q);
void AddQ(Queue* q, ElementType item);
int IsEmptyQ(Queue* q);
ElementType DeleteQ(Queue* q);

#endif